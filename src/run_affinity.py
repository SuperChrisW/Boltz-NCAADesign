from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import torch.nn.functional as F
from boltz.data import const
from .dataset_ncaa import NCAA_PredictionDataset, make_ncaa_loader, transfer_batch_to_device
from .template_constraints import LigandTemplateModule, LigandTemplateLoader
from .model_loader import create_ligand_template_module, load_template_coords
from .forward import trunk_forward, affinity_forward
from .io_utils import save_clipped_structure

def build_padded_coords(
    feats,
    dict_out,
    batch_ncaa,
    target_res_window: np.ndarray,                # 1-based residue indices to keep (ligand side)
    template_npz_path: Optional[Path] = None,
):
    """
    Args:
      target_res_window: 1-based residue indices (numpy array) to keep (applied to NONPOLYMER tokens)
      template_npz_path: optional output path where protein/ligand reference coordinates will be written
    Returns:
      padded_coords:  [K, N_kept_padded, 3]  (here K==1; padded to N_new_atom via token_to_rep_atom)
      dropped_coords: [K, N_kept, 3]         (true clipped coords before padding)
      kept_atom_mask: [N_atoms_kept] bool    (mask in atom space AFTER atom unpadding)
    """
        
    def _squeeze_batch_dim(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == 1:
            return t[0]
        return t

    # ------------------------
    # Unpad atom-level tensors
    # ------------------------
    atom_pad_mask = _squeeze_batch_dim(feats["atom_pad_mask"]).bool()                  # (N_atoms_total,)
    atom_to_token  = _squeeze_batch_dim(feats["atom_to_token"])                        # (N_atoms_total, N_tokens_pad) or [1,*,*]
    if atom_to_token.dim() == 3 and atom_to_token.shape[0] == 1:
        atom_to_token = atom_to_token[0]
    atom_to_token = atom_to_token[atom_pad_mask]                                       # (N_atoms, N_tokens_pad)

    coords = dict_out["coords"]                                                        # [B?, N_atoms_total, 3]
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords[atom_pad_mask]                                                     # (N_atoms, 3)

    device = coords.device

    # ------------------------
    # Token-level tensors
    # ------------------------
    token_pad_mask = _squeeze_batch_dim(batch_ncaa["token_pad_mask"]).bool()          # (N_tokens_pad,)
    mol_type       = _squeeze_batch_dim(batch_ncaa["mol_type"]).long()                # (N_tokens_pad,)
    token_res_idx  = _squeeze_batch_dim(batch_ncaa["residue_index"]).long()                 # (N_tokens_pad,)
    token_to_rep_atom = _squeeze_batch_dim(batch_ncaa["token_to_rep_atom"])           # (N_tokens_pad, N_new_atom)

    # Basic sanity checks
    #assert atom_to_token.shape[1] == token_pad_mask.shape[0], "atom_to_token and token_pad_mask mismatch"
    assert token_res_idx.shape[0] == token_pad_mask.shape[0], "res_idx and token_pad_mask mismatch"
    assert token_to_rep_atom.shape[0] == token_pad_mask.shape[0], "token_to_rep_atom and token_pad_mask mismatch"
    #assert N_token <= token_pad_mask.shape[0], "N_token exceeds token_pad_mask length"

    protein_token_mask = token_pad_mask & (mol_type == const.chain_type_ids["PROTEIN"])                         # (N_tokens_pad,)
    ligand_token_mask  = token_pad_mask & (mol_type == const.chain_type_ids["NONPOLYMER"])

    # ---------------------------------------------
    # Build atom masks via token membership (dense)
    # ---------------------------------------------
    atok = atom_to_token                                                # (N_atoms, N_token)
    N_prot_token = protein_token_mask.sum().item()
    protein_atom_mask = (atok[:, :N_prot_token]).sum(dim=1) > 0         # (N_atoms,)
    N_prot_atom = protein_atom_mask.sum().item()
    lig_atok = atok[N_prot_atom:, N_prot_token:]
    lig_atom_res_map = lig_atok.argmax(dim=-1)

    # ---------------------------------------------------
    # Keep: all protein atoms + ligand atoms in the window
    # ---------------------------------------------------
    # target_res_window is 1-based; token_res_idx is assumed 1-based.
    trg = torch.as_tensor(
        target_res_window,
        dtype=lig_atom_res_map.dtype,
        device=lig_atom_res_map.device,
    )
    lig_atom_res_map += 1 # convert to 1-th base
    local_lig_atom_mask = torch.isin(lig_atom_res_map, trg)

    # Pad local_lig_atom_mask at the top (prepend False) to match atom_to_token.shape[0]
    pad_len = atom_to_token.shape[0] - local_lig_atom_mask.shape[0]
    if pad_len < 0:
        raise ValueError("local_lig_atom_mask is longer than atom_to_token rows.")
    if pad_len > 0:
        global_lig_atom_mask = torch.cat([
            torch.zeros(pad_len, dtype=local_lig_atom_mask.dtype, device=local_lig_atom_mask.device),
            local_lig_atom_mask
        ], dim=0)
    else:
        global_lig_atom_mask = local_lig_atom_mask
    kept_atom_mask = (protein_atom_mask | global_lig_atom_mask)                  # (N_atoms,)
    dropped_coords = coords[kept_atom_mask]   

    N_new_atom = token_to_rep_atom.shape[-1]
    n_pad = N_new_atom - dropped_coords.shape[0]
    if n_pad < 0:
        raise ValueError(
            f"Requested padding target (N_new_atom={N_new_atom}) is smaller than kept atoms "
            f"(N_atoms_kept={dropped_coords.shape[0]}). Increase N_new_atom or adjust selection."
        )
    padded_coords = F.pad(dropped_coords, pad=(0, 0, 0, n_pad), value=0.0)            # (N_new_atom, 3)

    # ---------------------------------------
    # Optional: export reference coords (npz)
    # ---------------------------------------
    if template_npz_path is not None:
        p = Path(template_npz_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        protein_coords_np = coords[protein_atom_mask].detach().cpu().numpy()
        ligand_coords_np  = coords[global_lig_atom_mask].detach().cpu().numpy()

        protein_mask_np = np.ones(protein_coords_np.shape[0], dtype=bool)
        ligand_mask_np  = np.ones(ligand_coords_np.shape[0],  dtype=bool)

        np.savez(
            p,
            protein_coords=protein_coords_np,
            ligand_coords=ligand_coords_np,
            protein_mask=protein_mask_np,
            ligand_mask=ligand_mask_np,
        )
    return padded_coords.unsqueeze(0), dropped_coords.unsqueeze(0), kept_atom_mask

def pred_res_affinity_once(
    *,
    cfg,            # RunConfig object containing configuration parameters
    processed,      # Preprocessed data bundle with manifest, dirs, etc.
    feats,          # Feature dictionary input for structure/affinity prediction
    dict_out,       # Output dictionary from trunk_forward/structure model
    model_struct,   # Structure prediction model (Boltz2 or compatible)
    model_aff,      # Affinity prediction model (Boltz2 affinity head or compatible)
    device,         # PyTorch device (e.g., torch.device("cuda"))
    template_module,
    save_structure: bool = True,
):
    # ---------- Preparation ----------
    atom_mask = feats["atom_pad_mask"]
    N_atom, N_token = feats["atom_to_token"][atom_mask.bool(), ...].shape
    target_res_list = list(range(cfg.residue_min, cfg.residue_max + 1))

    # Prepare full coordinates array for dataset construction
    num_samples = dict_out['coords'].shape[0]
    if num_samples > 1 and 'iptm' in dict_out:
        best_sample_idx = torch.argmax(dict_out['iptm']).item()
    else:
        best_sample_idx = 0
    full_coords = dict_out['coords'][best_sample_idx].detach().cpu().numpy()  # [N_unpadded_atoms, 3]

    # Construct dataset/loader for affinity evaluation
    dataset = NCAA_PredictionDataset(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=cfg.mol_dir,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
        affinity=True,
        target_res_idx=target_res_list,
        tokenize_res_window=cfg.tokenize_res_window,
        max_tokens=cfg.affinity_max_tokens,
        max_atoms=cfg.affinity_max_atoms,
        trunk_coords=full_coords,
    )
    loader = make_ncaa_loader(dataset, batch_size=1, num_workers=2)

    # Prepare models and result storage
    model_struct.to(device).eval()
    model_aff.to(device).eval()
    template_module.to(device).eval() if template_module is not None else None
    results = {}

    # ---------- Main per-residue loop ----------
    for i, batch_ncaa in enumerate(loader):
        try:
            # 1. Determine 1-based target residue index and (potential) window
            if i >= len(target_res_list):
                print(f"[WARN] Dataset returned more batches ({i+1}) than expected residues ({len(target_res_list)}). Stopping.")
                break
            
            target_res_idx = target_res_list[i]
            if cfg.tokenize_res_window == 0:
                target_res_window = np.array([target_res_idx])  # single residue (1-based)
                window_start = target_res_idx
                window_end = target_res_idx
            else:
                window_start = max(1, target_res_idx - cfg.tokenize_res_window)
                window_end = target_res_idx + cfg.tokenize_res_window + 1  # inclusive
                target_res_window = np.arange(window_start, window_end)
            print(f"[INFO] Processing residue {target_res_idx}/{target_res_list[-1]} (window: {target_res_window}, batch_idx: {i})")

            # 2. Compute padded/dropped coords and atom mask for this target/region
            record = batch_ncaa["record"][0]
            struct_npz = processed.targets_dir / f"{record.id}.npz"
            template_npz = processed.template_dir / f"{record.id}_template_res{target_res_idx}.npz"
            padded_coords, dropped_coords, kept_atom_mask = build_padded_coords(
                feats,
                dict_out,
                batch_ncaa,
                target_res_window,
                template_npz_path=template_npz,
            )
            batch_ncaa = transfer_batch_to_device(batch_ncaa, device, i)
            padded_coords = padded_coords.to(device)
            pad_size = batch_ncaa["token_to_rep_atom"].shape[-1] - dropped_coords.shape[1]

            # 3. Compose paths and convert necessary objects for saving
            clipped_pdb = processed.template_dir / f"{record.id}_clipped_res{target_res_idx}.pdb"

            kept_mask_np = kept_atom_mask.detach().cpu().numpy()
            dropped_coords_np = dropped_coords.detach().cpu().numpy()

            # 4. Save clipped structure if requested
            if save_structure:
                save_clipped_structure(
                    struct_path=struct_npz,
                    out_pdb_path=clipped_pdb,
                    kept_atom_mask=kept_mask_np,
                    clipped_coords=dropped_coords_np,
                    boltz2=True,
                )
            #FIXME: revise load function, save the clipped structure as format for template_module
            if template_module is not None:
                reference_coords = template_module.load_reference_coords(
                    npz_path=template_npz,
                    protein_key="protein_coords",
                    ligand_key="ligand_coords",
                )
            else:
                reference_coords = None

            # 5. Run structure sampling/trunk and (optionally) save output structure
            out_trunk = trunk_forward(
                model_struct,
                batch_ncaa,
                recycling_steps=cfg.predict_args_affinity["recycling_steps"],
                num_sampling_steps=cfg.predict_args_affinity["sampling_steps"],
                diffusion_samples=cfg.predict_args_affinity["diffusion_samples"],
                max_parallel_samples=cfg.predict_args_affinity["max_parallel_samples"],
                ligand_template_module=template_module,
                reference_coords=reference_coords,
                run_confidence_sequentially=True,
            )

            masked_coords = out_trunk["sample_atom_coords"].detach().cpu().numpy()
            if save_structure:
                pred_pdb_path = processed.targets_dir / f"{record.id}_pred_res{window_start}-{window_end}.pdb"
                save_clipped_structure(
                    struct_path=struct_npz,
                    out_pdb_path=pred_pdb_path,
                    kept_atom_mask=kept_mask_np,
                    clipped_coords=masked_coords[:, :-pad_size, :],
                    boltz2=True,
                )

            # Substitute truncated/padded coords into trunk output for affinity head
            out_trunk["sample_atom_coords"] = padded_coords

            # 6. Affinity head (compute affinity probability and value)
            out_aff = affinity_forward(model_aff, batch_ncaa, out_trunk)
            results[target_res_idx] = {
                "affinity_probability_binary": out_aff["affinity_probability_binary"],
                "affinity_pred_value": out_aff["affinity_pred_value"],
            }
            print(f"[INFO] Successfully processed residue {target_res_idx}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process residue {target_res_list[i] if i < len(target_res_list) else 'unknown'}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next residue instead of stopping
            continue

    # ---------- Done ----------
    print(f"[INFO] Affinity prediction complete: {len(results)}/{len(target_res_list)} residues processed")
    if len(results) < len(target_res_list):
        missing = set(target_res_list) - set(results.keys())
        print(f"[WARN] Missing predictions for residues: {sorted(missing)}")
    return results
