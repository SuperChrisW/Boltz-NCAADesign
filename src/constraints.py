from __future__ import annotations
import numpy as np
from typing import Optional
#from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

def generate_contact_constraints_token_based(
    trunk_coords: np.ndarray,
    structure,
    atom_to_token: dict,
    token_data: np.ndarray,
    receptor_chain_asym_id: int = 0,
    ligand_chain_asym_id: int = 1,
    ligand_res_indices: np.ndarray | None = None,
    max_distance: float = 3.5,
    force: bool = False,
) -> list[tuple]:
    """
    Robust token-based contact constraints from coordinates.

    Returns a list of ((chain_id, local_token_idx), (chain_id, local_token_idx), max_distance, force).
    Local token indices are w.r.t. the receptor- or ligand-token *subarrays* returned by the masks below.
    """
    # ---- 0) Basic checks
    if trunk_coords.ndim != 2 or trunk_coords.shape[1] != 3:
        raise ValueError(f"trunk_coords should be [N_atoms, 3], got {trunk_coords.shape}")

    if not len(token_data):
        return []

    # ---- 1) Build receptor / ligand token masks
    rec_mask = (token_data["asym_id"] == receptor_chain_asym_id)
    lig_mask = (token_data["asym_id"] == ligand_chain_asym_id)

    # FIXME: mismatch between lig_token['res_idx'] and ligand_res_indices
    rec_tokens = token_data[rec_mask]
    lig_tokens = token_data[lig_mask]

    if len(rec_tokens) == 0 or len(lig_tokens) == 0:
        print("Warning: No receptor or ligand tokens found for contact constraints.")
        return []

    # ---- 2) Collect atom indices for each group from token slices (no ordering assumptions)
    def tokens_to_atom_indices(tokens) -> np.ndarray:
        if len(tokens) == 0:
            return np.empty((0,), dtype=int)
        # Concatenate [start, start+atom_num) for each token
        parts = [
            np.arange(int(tok["atom_idx"]), int(tok["atom_idx"]) + int(tok["atom_num"]), dtype=int)
            for tok in tokens
            if int(tok["atom_num"]) > 0
        ]
        return np.concatenate(parts) if parts else np.empty((0,), dtype=int)

    rec_atoms = tokens_to_atom_indices(rec_tokens)
    lig_atoms = tokens_to_atom_indices(lig_tokens)

    # Bound checks (atoms must exist in trunk_coords)
    n_atoms = trunk_coords.shape[0]
    if (rec_atoms >= n_atoms).any() or (lig_atoms >= n_atoms).any():
        raise IndexError("Token atom indices exceed trunk_coords length.")

    rec_coords = trunk_coords[rec_atoms]
    lig_coords = trunk_coords[lig_atoms]

    # ---- 3) Map each atom -> local token index *within its own group*
    # Build an array of length N_atoms set to -1 by default
    atom_to_local_rec = np.full(n_atoms, -1, dtype=int)
    atom_to_local_lig = np.full(n_atoms, -1, dtype=int)

    # Fill with local indices: for token t (local), mark all its atoms with t
    for local_t, tok in enumerate(rec_tokens):
        start = int(tok["atom_idx"])
        num = int(tok["atom_num"])
        if num > 0:
            atom_to_local_rec[start:start+num] = local_t

    for local_t, tok in enumerate(lig_tokens):
        start = int(tok["atom_idx"])
        num = int(tok["atom_num"])
        if num > 0:
            atom_to_local_lig[start:start+num] = local_t

    # ---- 4) Find atom–atom neighbors within max_distance
    close_pairs_token = set()
    tree = cKDTree(lig_coords)
    # query_ball_point returns list of neighbor indices in lig_coords for each rec atom
    neighbors = tree.query_ball_point(rec_coords, r=max_distance)
    for i_rec_local_atom, lig_neighbor_list in enumerate(neighbors):
        if not lig_neighbor_list:
            continue
        rec_atom_idx = rec_atoms[i_rec_local_atom]
        rec_local_token = atom_to_local_rec[rec_atom_idx]
        if rec_local_token < 0:
            continue
        for j_lig_local_atom in lig_neighbor_list:
            lig_atom_idx = lig_atoms[j_lig_local_atom]
            lig_local_token = atom_to_local_lig[lig_atom_idx]
            if lig_local_token < 0:
                continue
            close_pairs_token.add((rec_local_token, lig_local_token))

    # ---- 5) Build constraint tuples (deduplicated)
    contact_constraints: list[tuple] = []
    for rec_t, lig_t in sorted(close_pairs_token):
        contact_constraints.append(
            ((receptor_chain_asym_id, int(rec_t)),
             (ligand_chain_asym_id, int(lig_t)),
             float(max_distance),
             bool(force))
        )
    
    # ---- 6) Logging
    print(f"Generated {len(contact_constraints)} contact constraints (token-based)")
    print(f"  - Receptor tokens: {len(rec_tokens)} (atoms: {len(rec_atoms)})")
    print(f"  - Ligand tokens:   {len(lig_tokens)} (atoms: {len(lig_atoms)})")
    print(f"  - Max distance:    {max_distance:.2f} Å")
    print(f"  - Contact:    {contact_constraints}")

    return contact_constraints