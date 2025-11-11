from __future__ import annotations
import torch
import numpy as np
import torch.nn.functional as F
from .dataset_ncaa import NCAA_PredictionDataset, make_ncaa_loader, transfer_batch_to_device
from .template_constraints import LigandTemplateModule, LigandTemplateLoader
from .model_loader import create_ligand_template_module, load_template_coords
from .forward import trunk_forward, affinity_forward
from .io_utils import save_clipped_structure

def build_padded_coords(feats, dict_out, batch_ncaa, N_token, target_res_window: np.ndarray):
    '''
    Args:
      target_res_window: 1-based residue indices (numpy array) to keep
    Returns:
      padded_coords:  [K, N_kept_padded, 3]
      dropped_coords: [K, N_kept, 3]        (pre-pad, true clipped coords)
      kept_atom_mask: [N_atoms_masked] bool mask of kept atoms (before padding)
    '''

    atom_mask = feats["atom_pad_mask"]
    correct_atom_to_token = feats["atom_to_token"][atom_mask.bool(), ...]
    print("DEBUG atom_mask shape:", atom_mask.shape)
    print("DEBUG correct_atom_to_token shape:", correct_atom_to_token.shape)
    print("DEBUG dict_out['coords'] shape:", dict_out["coords"].shape)
    correct_coords = dict_out["coords"][:, atom_mask.squeeze(0).bool(), ...]

    rec_token_mask = batch_ncaa["mol_type"] == 0 #receptor chain token mask
    lig_token_num = N_token - torch.sum(rec_token_mask, dim=1) # lig atom number
    # Extract scalar value if tensor (batch_size=1)
    if isinstance(lig_token_num, torch.Tensor):
        lig_token_num = lig_token_num.item() if lig_token_num.numel() == 1 else lig_token_num[0].item()
    lig_token_mask = torch.zeros(lig_token_num, dtype=torch.bool)
    # target_res_window is 1-based, convert to 0-based for array indexing
    target_res_0based = target_res_window - 1
    # Ensure indices are valid (within bounds and >= 0)
    valid_mask = (target_res_0based >= 0) & (target_res_0based < lig_token_num)
    if valid_mask.any():
        lig_token_mask[target_res_0based[valid_mask]] = True
    kept_token_mask = torch.concatenate((rec_token_mask[rec_token_mask == 1], lig_token_mask))
    kept_atom_mask = correct_atom_to_token.to(kept_token_mask.device) @ kept_token_mask.to(torch.long)
    dropped_coords = correct_coords[:, kept_atom_mask.bool(), :]

    pad_size = (0, 0, 0, batch_ncaa["token_to_rep_atom"].shape[-1] - dropped_coords.shape[1])
    padded_coords = F.pad(dropped_coords, pad_size, value=0.0)
    return padded_coords, dropped_coords, kept_atom_mask

def pred_res_affinity_once(
    *,
    cfg,            # RunConfig object containing configuration parameters
    processed,      # Preprocessed data bundle with manifest, dirs, etc.
    feats,          # Feature dictionary input for structure/affinity prediction
    dict_out,       # Output dictionary from trunk_forward/structure model
    model_struct,   # Structure prediction model (Boltz2 or compatible)
    model_aff,      # Affinity prediction model (Boltz2 affinity head or compatible)
    device,         # PyTorch device (e.g., torch.device("cuda"))
    save_structure: bool = True,
):
    # ---------- Preparation ----------
    atom_mask = feats["atom_pad_mask"]
    N_atom, N_token = feats["atom_to_token"][atom_mask.bool(), ...].shape

    # Prepare list of target residue indices (1-based)
    # Ensure we have valid residue range (should be set by caller if auto-detect was needed)
    if cfg.residue_min < 1 or cfg.residue_max < 1:
        raise ValueError(f"Invalid residue range: residue_min={cfg.residue_min}, residue_max={cfg.residue_max}. Both must be >= 1.")
    
    if cfg.residue_min == cfg.residue_max:
        target_res_list = [cfg.residue_min]
    else:
        target_res_list = list(range(cfg.residue_min, cfg.residue_max + 1))
    
    if len(target_res_list) == 0:
        raise ValueError(f"Empty residue range: residue_min={cfg.residue_min}, residue_max={cfg.residue_max}")
    
    print(f"[INFO] Affinity prediction: Processing {len(target_res_list)} residues: {target_res_list[0]} to {target_res_list[-1]}")

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
            else:
                window_start = max(1, target_res_idx - cfg.tokenize_res_window)
                window_end = target_res_idx + cfg.tokenize_res_window + 1  # inclusive
                target_res_window = np.arange(window_start, window_end)
            print(f"[INFO] Processing residue {target_res_idx}/{target_res_list[-1]} (window: {target_res_window}, batch_idx: {i})")

            # 2. Compute padded/dropped coords and atom mask for this target/region
            padded_coords, dropped_coords, kept_atom_mask = build_padded_coords(
                feats, dict_out, batch_ncaa, N_token, target_res_window
            )
            batch_ncaa = transfer_batch_to_device(batch_ncaa, device, i)
            padded_coords = padded_coords.to(device)
            pad_size = batch_ncaa["token_to_rep_atom"].shape[-1] - dropped_coords.shape[1]

            # 3. Compose paths and convert necessary objects for saving
            record = batch_ncaa["record"][0]
            struct_npz = processed.targets_dir / f"{record.id}.npz"
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

            # 5. Run structure sampling/trunk and (optionally) save output structure
            out_trunk = trunk_forward(
                model_struct,
                batch_ncaa,
                recycling_steps=cfg.predict_args_affinity["recycling_steps"],
                num_sampling_steps=cfg.predict_args_affinity["sampling_steps"],
                diffusion_samples=cfg.predict_args_affinity["diffusion_samples"],
                max_parallel_samples=cfg.predict_args_affinity["max_parallel_samples"],
                run_confidence_sequentially=True,
            )

            masked_coords = out_trunk["sample_atom_coords"].detach().cpu().numpy()
            if save_structure:
                pred_pdb_path = processed.targets_dir / f"{record.id}_pred_res{target_res_idx}.pdb"
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
