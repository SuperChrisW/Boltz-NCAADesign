from __future__ import annotations
import json
import numpy as np
import torch
from dataclasses import asdict, replace
from pathlib import Path
from typing import List
from boltz.data.types import StructureV2, Interface, Coords, Record, AffinityInfo
from boltz.data.write.pdb import to_pdb

def write_predictions(
    predictions: dict, records: List[Record], data_dir: Path, output_dir: Path,
    output_format: str = "pdb", boltz2: bool = True, write_embeddings: bool = False
) -> None:
    if predictions.get("exception", False):
        return

    coords, masks = predictions["coords"], predictions["masks"]
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)

    if "confidence_score" in predictions:
        argsort = torch.argsort(predictions["confidence_score"], descending=True)
        idx_to_rank = {int(idx): int(rank) for rank, idx in enumerate(argsort)}
    else:
        idx_to_rank = {i: i for i in range(coords.shape[0])}

    for record in records:
        struct = StructureV2.load(data_dir / f"{record.id}.npz")
        if hasattr(struct, "mask"):
            chain_map = {}
            for i, m in enumerate(struct.mask):
                if m:
                    chain_map[len(chain_map)] = i
        else:
            chain_map = {}
        struct = struct.remove_invalid_chains()

        out_dir = output_dir / record.id
        out_dir.mkdir(parents=True, exist_ok=True)

        has_plddt, has_pae, has_pde = ("plddt" in predictions), ("pae" in predictions), ("pde" in predictions)

        for model_idx in range(coords.shape[0]):
            model_coords = coords[model_idx]
            pad_mask = masks[model_idx].bool()

            coord_unpad = model_coords[pad_mask]
            coord_np = coord_unpad.detach().cpu().numpy()

            atoms = struct.atoms
            atoms["coords"] = coord_np
            atoms["is_present"] = True

            residues = struct.residues
            residues["is_present"] = True

            coord_wrapped = np.array([(xyz,) for xyz in coord_np], dtype=Coords)
            new_struct = replace(struct, atoms=atoms, residues=residues, interfaces=np.array([], dtype=Interface), coords=coord_wrapped)

            rank = idx_to_rank[model_idx]
            name = f"{record.id}_model_{rank}"

            if output_format == "pdb":
                with (out_dir / f"{name}.pdb").open("w") as f:
                    f.write(to_pdb(new_struct, plddts=predictions["plddt"][model_idx] if has_plddt else None, boltz2=boltz2))
            elif output_format == "mmcif":
                from boltz.data.write.mmcif import to_mmcif
                with (out_dir / f"{name}.cif").open("w") as f:
                    f.write(to_mmcif(new_struct, plddts=predictions["plddt"][model_idx] if has_plddt else None, boltz2=boltz2))
            else:
                np.savez_compressed(out_dir / f"{name}.npz", **asdict(new_struct))

            if boltz2 and rank == 0:
                np.savez_compressed(out_dir / f"pre_affinity_{record.id}.npz", **asdict(new_struct))

            if "confidence_score" in predictions:
                conf = {
                    "confidence_score": predictions["confidence_score"][model_idx].item(),
                    "ptm": predictions.get("ptm", torch.tensor(float("nan")))[model_idx].item() if "ptm" in predictions else None,
                    "iptm": predictions.get("iptm", torch.tensor(float("nan")))[model_idx].item() if "iptm" in predictions else None,
                    "ligand_iptm": predictions.get("ligand_iptm", torch.tensor(float("nan")))[model_idx].item() if "ligand_iptm" in predictions else None,
                    "protein_iptm": predictions.get("protein_iptm", torch.tensor(float("nan")))[model_idx].item() if "protein_iptm" in predictions else None,
                    "complex_plddt": predictions.get("complex_plddt", torch.tensor(float("nan")))[model_idx].item() if "complex_plddt" in predictions else None,
                    "complex_iplddt": predictions.get("complex_iplddt", torch.tensor(float("nan")))[model_idx].item() if "complex_iplddt" in predictions else None,
                    "complex_pde": predictions.get("complex_pde", torch.tensor(float("nan")))[model_idx].item() if "complex_pde" in predictions else None,
                    "complex_ipde": predictions.get("complex_ipde", torch.tensor(float("nan")))[model_idx].item() if "complex_ipde" in predictions else None,
                }
                with (out_dir / f"confidence_{record.id}_model_{rank}.json").open("w") as f:
                    json.dump(conf, f, indent=4)

            if has_plddt:
                np.savez_compressed(out_dir / f"plddt_{record.id}_model_{rank}.npz", plddt=predictions["plddt"][model_idx].detach().cpu().numpy())
            if has_pae:
                np.savez_compressed(out_dir / f"pae_{record.id}_model_{rank}.npz", pae=predictions["pae"][model_idx].detach().cpu().numpy())
            if has_pde:
                np.savez_compressed(out_dir / f"pde_{record.id}_model_{rank}.npz", pde=predictions["pde"][model_idx].detach().cpu().numpy())

def set_record_affinity(record, chain_id: int):
    from boltz.data.types import AffinityInfo
    return replace(record, affinity=AffinityInfo(chain_id=chain_id, mw=None))

def save_clipped_structure(
    *,
    struct_path: Path,          # path to {record.id}.npz or pre_affinity npz
    out_pdb_path: Path,         # where to write clipped PDB
    kept_atom_mask: np.ndarray, # shape [N_atoms] bool over the UNPADDED atom dimension used by feats
    clipped_coords: np.ndarray, # shape [K, N_kept, 3] or [N_kept, 3]
    boltz2: bool = True,
) -> None:
    """
    Create a new StructureV2 with only the kept atoms and residues, using the given clipped coords.
    Assumes kept_atom_mask refers to the same atom indexing used when unpadding the model coords.

    Notes:
      - We recompute per-residue atom_idx/atom_num by intersecting kept atoms with residue ranges.
      - Residues with zero kept atoms are dropped.
      - Chains are compacted similarly (res_idx/res_num recomputed).
    """

    structure = StructureV2.load(struct_path)
    structure = structure.remove_invalid_chains()

    # Normalize coords to [N_kept, 3]
    if clipped_coords.ndim == 3:
        # take first K if multiple; adjust if you need rank selection
        clipped_coords = np.asarray(clipped_coords[0])
    else:
        clipped_coords = np.asarray(clipped_coords)

    kept_atom_mask = np.asarray(kept_atom_mask, dtype=bool)
    assert kept_atom_mask.ndim == 1, "kept_atom_mask must be 1-D"
    n_kept = int(kept_atom_mask.sum())
    if n_kept != clipped_coords.shape[0]:
        raise ValueError(
            f"Mismatch: kept atoms {n_kept} vs coords {clipped_coords.shape[0]}"
        )

    # Slice atoms
    atoms_old = structure.atoms
    atoms_kept = atoms_old[kept_atom_mask].copy()
    atoms_kept["coords"] = clipped_coords
    atoms_kept["is_present"] = True

    # Build a global mapping old_atom_idx -> new_atom_idx (for residue reindex)
    old_to_new = -np.ones(len(atoms_old), dtype=int)
    old_to_new[np.where(kept_atom_mask)[0]] = np.arange(n_kept, dtype=int)

    # Recompute residues: keep residues with >=1 kept atom
    residues_old = structure.residues
    residues_new_list = []
    new_atom_cursor = 0

    for r in residues_old:
        a0 = int(r["atom_idx"])
        aN = a0 + int(r["atom_num"])
        # kept atoms within this residue range (global indices)
        kept_local = kept_atom_mask[a0:aN]
        count = int(kept_local.sum())
        if count == 0:
            continue

        # new residue record with updated atom range
        r_new = r.copy()
        r_new["atom_idx"] = new_atom_cursor
        r_new["atom_num"] = count
        r_new["is_present"] = True
        residues_new_list.append(r_new)
        new_atom_cursor += count

    residues_new = np.array(residues_new_list, dtype=residues_old.dtype)

    # Wrap coords into Coords dtype
    coord_wrapped = np.array([(xyz,) for xyz in clipped_coords], dtype=Coords)

    new_structure = replace(
        structure,
        atoms=atoms_kept,
        residues=residues_new,
        #chains=chains_new,
        coords=coord_wrapped,
        interfaces=np.array([], dtype=Interface),
    )

    # Write PDB
    out_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb_path, "w") as f:
        f.write(to_pdb(new_structure, plddts=None, boltz2=boltz2))
