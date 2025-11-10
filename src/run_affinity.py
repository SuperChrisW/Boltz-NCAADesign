from __future__ import annotations
import torch
import numpy as np
import torch.nn.functional as F
from .dataset_ncaa import NCAA_PredictionDataset, make_ncaa_loader, transfer_batch_to_device
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

def pred_res_affinity_once(*, cfg, processed, feats, dict_out, model_struct, model_aff, device):
    atom_mask = feats["atom_pad_mask"]
    N_atom, N_token = feats["atom_to_token"][atom_mask.bool(), ...].shape

    # 1-based index list (residue_min and residue_max are 1-based)
    if cfg.residue_min == cfg.residue_max:
        target_res_list = [cfg.residue_min]
    else:
        target_res_list = list(range(cfg.residue_min, cfg.residue_max + 1))

    atom_mask = feats["atom_pad_mask"].squeeze(0).bool().cpu().numpy()  # [N_total_atoms] padded at the end
    num_samples = dict_out['coords'].shape[0]
    if num_samples > 1 and 'iptm' in dict_out:
        best_sample_idx = torch.argmax(dict_out['iptm']).item()
    else:
        best_sample_idx = 0
    full_coords = dict_out['coords'][best_sample_idx].detach().cpu().numpy()  # [N_unpadded_atoms, 3]
    
    dataset = NCAA_PredictionDataset(
        manifest=processed.manifest, target_dir=processed.targets_dir, msa_dir=processed.msa_dir, mol_dir=cfg.mol_dir,
        constraints_dir=processed.constraints_dir, template_dir=processed.template_dir, extra_mols_dir=processed.extra_mols_dir,
        override_method=None, affinity=True, target_res_idx=target_res_list, tokenize_res_window=cfg.tokenize_res_window, 
        max_tokens=cfg.affinity_max_tokens, max_atoms=cfg.affinity_max_atoms, trunk_coords=full_coords,
    )
    loader = make_ncaa_loader(dataset, batch_size=1, num_workers=2)

    results = {}
    model_struct.to(device).eval()
    model_aff.to(device).eval()

    for i, batch_ncaa in enumerate(loader):
        target_res_idx = target_res_list[i]  # 1-based residue index
        if cfg.tokenize_res_window==0:
            target_res_window = np.array([target_res_idx])  # 1-based
        else:
            # Create 1-based window around target residue (inclusive)
            # Ensure minimum is at least 1 (1-based indexing)
            window_start = max(1, target_res_idx - cfg.tokenize_res_window)
            window_end = target_res_idx + cfg.tokenize_res_window + 1  # +1 for inclusive end
            target_res_window = np.array(list(range(window_start, window_end)))
        print("target_res_idx (1-based):", target_res_idx, "target_res_window (1-based):", target_res_window, "index:", i)

        padded_coords, dropped_coords, kept_atom_mask = build_padded_coords(feats, dict_out, batch_ncaa, N_token, target_res_window)
        batch_ncaa = transfer_batch_to_device(batch_ncaa, device, i)
        padded_coords.to(device)
        pad_size = batch_ncaa["token_to_rep_atom"].shape[-1] - dropped_coords.shape[1]

        # save clipped coords
        record = batch_ncaa["record"][0]
        struct_npz = (processed.targets_dir / f"{record.id}.npz")  # same as used elsewhere
        clipped_pdb = processed.template_dir / f"{record.id}_clipped_res{target_res_idx}.pdb"

        # Convert tensors to numpy
        kept_mask_np = kept_atom_mask.detach().cpu().numpy()
        dropped_coords_np = dropped_coords.detach().cpu().numpy()  # [K, N_kept, 3] or [N_kept, 3]

        save_clipped_structure(
            struct_path=struct_npz,
            out_pdb_path=clipped_pdb,
            kept_atom_mask=kept_mask_np,
            clipped_coords=dropped_coords_np,
            boltz2=True,
        )
        # end

        #FIXME: batch_ncaa token
        out_trunk = trunk_forward(
            model_struct, batch_ncaa,
            recycling_steps=cfg.predict_args_affinity["recycling_steps"],
            num_sampling_steps=cfg.predict_args_affinity["sampling_steps"],
            diffusion_samples=cfg.predict_args_affinity["diffusion_samples"],
            max_parallel_samples=cfg.predict_args_affinity["max_parallel_samples"],
            run_confidence_sequentially=True,
        )
        # save prediction protein-ligand strucure for check
        masked_coords = out_trunk["sample_atom_coords"].detach().cpu().numpy()
        save_clipped_structure(
            struct_path=struct_npz,
            out_pdb_path= processed.targets_dir / f"{record.id}_pred_res{target_res_idx}.pdb",
            kept_atom_mask=kept_mask_np,
            clipped_coords=masked_coords[:,:-pad_size,:],
            boltz2=True,
        )
        out_trunk["sample_atom_coords"] = padded_coords
        out_aff = affinity_forward(model_aff, batch_ncaa, out_trunk)

        results[target_res_idx] = {
            "affinity_probability_binary": out_aff["affinity_probability_binary"],
            "affinity_pred_value": out_aff["affinity_pred_value"],
        }

    return results
