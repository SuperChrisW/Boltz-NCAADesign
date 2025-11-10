from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Tuple, Optional

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors

from .constraints import generate_contact_constraints_token_based

from boltz.data import const
from boltz.data.tokenize.boltz2 import tokenize_structure, Boltz2Tokenizer
from boltz.data.mol import load_canonicals
from boltz.data.parse.schema import *
from boltz.data.types import (
    AffinityInfo,
    ChainInfo,
    StructureV2,
    TokenBondV2,
    Tokenized,
    TokenV2,
    Chain,
    Residue,
    AtomV2,
    BondV2,
    ChiralAtomConstraint,
    Connection,
    Coords,
    Ensemble,
    InferenceOptions,
    Interface,
    PlanarBondConstraint,
    PlanarRing5Constraint,
    PlanarRing6Constraint,
    RDKitBoundsConstraint,
    Record,
    ResidueConstraints,
    StereoBondConstraint,
    Structure,
    StructureInfo,
    StructureV2,
    Target,
    TemplateInfo,
)
# Reuse helper: get attribute or dict-like key
def _get(obj, name):
    if hasattr(obj, name): return getattr(obj, name)
    try: return obj[name]
    except Exception: return asdict(obj)[name]

class NCAA_Tokenizer(Boltz2Tokenizer):
    """Tokenizer that atomizes a window around a target residue on the ligand chain and rebuilds bonds."""
    def __init__(self, mol_dir: Optional[Path] = None):
        super().__init__()
        self.mol_dir = Path(mol_dir) if mol_dir is not None else None
        self._refmol_cache = load_canonicals(mol_dir) if mol_dir is not None else None #limited to canonicals AAs

    def NCAA_tokenize(self, input_data, res_id: int, tokenize_res_window: int, trunk_coords: np.ndarray = None):
        lig_chain = input_data.structure.chains[1]       # FIXME: extract the protein receptor chains. put ligand chain at the end

        (
            token_data,
            token_bonds,
            struct_data,
            residue_constraints,
            atom_to_token,
            record,
        ) = self.atomize_residues(
            target=input_data,
            ligand_chain=lig_chain,
            target_res_id=res_id-1,
            tokenize_res_window=tokenize_res_window,
            trunk_coords=trunk_coords,
        )

        # Generate contact constraints from trunk coordinates
        contact_constraints = []
        if trunk_coords is not None:
            # Determine ligand residue indices (1-based)
            if tokenize_res_window == 0:
                ligand_res_indices = np.array([res_id])
            else:
                window_start = max(1, res_id - tokenize_res_window)
                window_end = res_id + tokenize_res_window + 1
                ligand_res_indices = np.array(list(range(window_start, window_end)))
            
            contact_constraints = generate_contact_constraints_token_based(
                trunk_coords=trunk_coords,
                structure=input_data.structure,
                atom_to_token=atom_to_token,
                token_data=token_data,
                receptor_chain_asym_id=0,  # FIXME: should be configurable
                ligand_chain_asym_id=1,    # FIXME: should be configurable
                ligand_res_indices=ligand_res_indices,
                max_distance=3.5,
                force=True,
            )

        # Templates (unchanged)
        if input_data.templates is not None:
            template_tokens, template_bonds = {}, {}
            for tmpl_id, tmpl in input_data.templates.items():
                td, tb = tokenize_structure(tmpl)
                template_tokens[tmpl_id] = td
                template_bonds[tmpl_id] = tb
        else:
            template_tokens = template_bonds = None

        return Tokenized(
            tokens=token_data, # New
            bonds=token_bonds, # New
            structure=struct_data, # New
            msa={}, #FIXME:
            record= record, # New
            residue_constraints=residue_constraints, # New
            templates=input_data.templates, #FIXME:
            template_tokens=template_tokens, #FIXME:
            template_bonds=template_bonds, #FIXME:
            extra_mols=input_data.extra_mols, #FIXME:
        ), contact_constraints

    def atomize_residues(
        self,
        target: Target,
        ligand_chain: Chain,
        target_res_id: int, # 0-th base index
        tokenize_res_window: int = 0,
        trunk_coords: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, StructureV2, ResidueConstraints, dict, Record]:
        """
        Atomize residues in the ligand chain and process all chains.
        
        Returns:
            token_data: Array of token data
            token_bonds: Array of token bonds
            struct_data: StructureV2 object
            residue_constraints: ResidueConstraints object
            atom_to_token: Mapping from atom index to token index
            record: Record metadata
        """
        struct = target.structure
        lig_asym_id = _get(ligand_chain, "asym_id") # FIXME: assume to be 1
        lig_res_start, lig_res_num = _get(ligand_chain, "res_idx"), _get(ligand_chain, "res_num")
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        
        # Get ligand residues and calculate window to keep
        ligand_res = struct.residues[lig_res_start : lig_res_start + lig_res_num]
        ligand_res_indices = np.array([res["res_idx"] for res in ligand_res])
        
        if target_res_id not in ligand_res_indices:
            raise ValueError(f"Residue {target_res_id} not in ligand chain {ligand_chain.chain_id}")
        
        # FIXME: not consider incontinuous peptide case
        target_pos = np.where(ligand_res_indices == target_res_id)[0][0]
        start = max(0, target_pos - tokenize_res_window)
        end = min(len(ligand_res_indices), target_pos + tokenize_res_window + 1)
        kept_residues = ligand_res[start:end]

        local_to_global_res_idx = {local_idx: kept_res["res_idx"] 
                                   for local_idx, kept_res in enumerate(kept_residues)}
        # Initialize data structures
        glob_idx_map = {}  # (chain_asym_id, res_idx, atom_name) -> global_atom_idx
        atom_to_token = {}  # global_atom_idx -> token_idx
        token_data = []
        atom_data_raw = []  # Store raw atom data: (name, element, charge, coords, conformer, is_present, chirality)
        res_data = []
        chain_data = []
        
        # Track indices
        global_atom_idx = 0
        global_res_idx = 0
        token_idx = 0
        
        # ========================================================================
        # STEP 1: Process protein receptor chains (non-ligand chains)
        # ========================================================================
        for chain in struct.chains:
            if chain["asym_id"] == lig_asym_id:
                continue  # Skip ligand chain, process it later
            
            chain_atom_start = global_atom_idx
            chain_res_start = global_res_idx
            
            res_start = chain['res_idx']
            res_num = chain['res_num']
            
            for res in struct.residues[res_start:res_start+res_num]:
                if not res["is_standard"]:
                    continue
                
                atom_start, atom_end = res["atom_idx"], res["atom_idx"] + res["atom_num"]
                residue_atom_start = global_atom_idx
                
                # Get center and disto atom coordinates
                if trunk_coords is not None:
                    c_coords = trunk_coords[res["atom_center"]].copy()
                    d_coords = trunk_coords[res["atom_disto"]].copy()
                    if (c_coords == 0).all():
                        c_coords = struct.coords[res["atom_center"]]["coords"]
                    if (d_coords == 0).all():
                        d_coords = struct.coords[res["atom_disto"]]["coords"]
                else:
                    c_coords = struct.coords[res["atom_center"]]["coords"]
                    d_coords = struct.coords[res["atom_disto"]]["coords"]

                # Compute frame for protein residues
                frame_rot, frame_t, frame_mask = np.eye(3).flatten(), np.zeros(3), False
                atoms = struct.atoms[atom_start:atom_end]
                if len(atoms) >= 3:
                    n, ca, c = atoms[0], atoms[1], atoms[2]
                    if trunk_coords is not None:
                        n_coords = trunk_coords[atom_start] if not (trunk_coords[atom_start] == 0).all() else n["coords"]
                        ca_coords = trunk_coords[atom_start + 1] if not (trunk_coords[atom_start + 1] == 0).all() else ca["coords"]
                        c_coords_frame = trunk_coords[atom_start + 2] if not (trunk_coords[atom_start + 2] == 0).all() else c["coords"]
                    else:
                        n_coords = n["coords"]
                        ca_coords = ca["coords"]
                        c_coords_frame = c["coords"]
                    
                    frame_mask = ca["is_present"] & c["is_present"] & n["is_present"]
                    if frame_mask:
                        from boltz.data.tokenize.boltz2 import compute_frame
                        r, t = compute_frame(n_coords, ca_coords, c_coords_frame)
                        frame_rot, frame_t = r.flatten(), t

                is_present = res["is_present"] & struct.atoms[res["atom_center"]]["is_present"]
                is_disto_present = res["is_present"] & struct.atoms[res["atom_disto"]]["is_present"]

                # Add token for this residue
                token_data.append((
                    token_idx,  # token_idx
                    residue_atom_start,  # atom_idx (first atom in residue)
                    res["atom_num"],  # atom_num
                    global_res_idx,  # res_idx
                    res["res_type"],  # res_type
                    res["name"],  # res_name
                    chain["sym_id"],  # sym_id
                    chain["asym_id"],  # asym_id
                    chain["entity_id"],  # entity_id
                    chain["mol_type"],  # mol_type
                    residue_atom_start + (res["atom_center"] - atom_start),  # center_idx (relative to residue start)
                    residue_atom_start + (res["atom_disto"] - atom_start),  # disto_idx (relative to residue start)
                    c_coords,  # center_coords
                    d_coords,  # disto_coords
                    is_present,  # resolved_mask
                    is_disto_present,  # disto_mask
                    False,  # modified
                    frame_rot,  # frame_rot
                    frame_t,  # frame_t
                    bool(frame_mask),  # frame_mask
                    chain["cyclic_period"],  # cyclic_period
                    False  # affinity_mask
                ))

                # Add all atoms in this residue
                for atom in struct.atoms[atom_start:atom_end]:
                    atom_name = atom["name"]
                    glob_idx_map[(chain['asym_id'], res['res_idx'], atom_name)] = global_atom_idx
                    atom_to_token[global_atom_idx] = token_idx
                    
                    atom_data_raw.append((
                        atom["name"],
                        #Chem.GetPeriodicTable().GetAtomicNumber(atom["name"]),
                        #atom["charge"],
                        atom["coords"],
                        #atom["conformer"],
                        atom["is_present"],
                        #atom["chirality"],
                    ))
                    global_atom_idx += 1

                # Add residue data
                residue_center_idx = residue_atom_start + (res["atom_center"] - atom_start)
                residue_disto_idx = residue_atom_start + (res["atom_disto"] - atom_start)
                res_data.append((
                    res["name"],
                    const.token_ids[res["name"]],
                    global_res_idx,
                    residue_atom_start,  # atom_idx (first atom in residue)
                    res["atom_num"],
                    residue_center_idx,  # atom_center
                    residue_disto_idx,  # atom_disto
                    res["is_standard"],
                    res["is_present"],
                ))
                
                token_idx += 1
                global_res_idx += 1

            # Add chain data
            chain_data.append((
                chain["name"],
                const.chain_type_ids["PROTEIN"],
                chain["entity_id"],
                chain["sym_id"],
                chain["asym_id"],
                chain_atom_start,  # atom_idx
                global_atom_idx - chain_atom_start,  # atom_num
                chain_res_start,  # res_idx
                global_res_idx - chain_res_start,  # res_num
                chain["cyclic_period"],
            ))

        # ========================================================================
        # STEP 2: Process ligand chain (atomized residues)
        # ========================================================================
        lig_chain = None
        for chain in struct.chains:
            if chain["asym_id"] == lig_asym_id:
                lig_chain = chain
                break
        if lig_chain is None:
            raise ValueError(f"Ligand chain {lig_asym_id} not found")
        
        lig_chain_atom_start = global_atom_idx
        lig_chain_res_start = global_res_idx
        
        for local_res_idx, res in enumerate(kept_residues):
            glob_res_idx = res["res_idx"]  # Keep global index for glob_idx_map lookups
            atom_start, atom_end = res["atom_idx"], res["atom_idx"] + res["atom_num"]
            residue_atom_start = global_atom_idx
            
            # Get atom coordinates from trunk_coords if available
            if trunk_coords is not None:
                atom_coords = trunk_coords[atom_start:atom_end].copy()
                zero_mask = (atom_coords == 0).all(axis=1)
                if zero_mask.any():
                    original_coords = struct.coords[atom_start:atom_end]["coords"]
                    atom_coords[zero_mask] = original_coords[zero_mask]
            else:
                atom_coords = struct.coords[atom_start:atom_end]["coords"]

            atoms = struct.atoms[atom_start:atom_end]
            for j, atom in enumerate(atoms):
                atom_name = atom["name"]
                is_present = res["is_present"] & atom["is_present"]
                
                # Add token for each atom (atomized)
                token_data.append((
                    token_idx,  # token_idx
                    global_atom_idx,  # atom_idx
                    1,  # atom_num
                    local_res_idx,  # res_idx (use local index for MSA compatibility)
                    unk_prot_id,  # res_type
                    res["name"],  # res_name
                    lig_chain["sym_id"],  # sym_id
                    lig_chain["asym_id"],  # asym_id
                    lig_chain["entity_id"],  # entity_id
                    const.chain_type_ids["NONPOLYMER"],  # mol_type
                    global_atom_idx,  # center_idx
                    global_atom_idx,  # disto_idx
                    atom_coords[j],  # center_coords
                    atom_coords[j],  # disto_coords
                    is_present,  # resolved_mask
                    is_present,  # disto_mask
                    True,  # modified
                    np.eye(3).flatten(),  # frame_rot
                    np.zeros(3),  # frame_t
                    False,  # frame_mask
                    lig_chain["cyclic_period"],  # cyclic_period
                    True,  # affinity_mask
                ))
                
                atom_data_raw.append((
                    atom_name,
                    #atom.element,
                    #atom.charge,
                    atom_coords[j],
                    #atom.conformer,
                    is_present,
                    #atom.chirality,
                ))

                # Use global res_idx for glob_idx_map (needed for constraint lookups)
                glob_idx_map[(lig_chain['asym_id'], glob_res_idx, atom_name)] = global_atom_idx
                atom_to_token[global_atom_idx] = token_idx
                token_idx += 1
                global_atom_idx += 1

            # Add residue data for ligand (each atom is a token, but residue still exists)
            res_data.append((
                res["name"],
                unk_prot_id, 
                local_res_idx,
                residue_atom_start,  # atom_idx
                res["atom_num"],
                residue_atom_start,  # atom_center (use first atom)
                residue_atom_start,  # atom_disto (use first atom)
                False,
                res["is_present"],
            ))
            global_res_idx += 1

        # Add ligand chain data
        lig_chain_atom_num = global_atom_idx - lig_chain_atom_start
        lig_chain_res_num = global_res_idx - lig_chain_res_start
        chain_data.append((
            lig_chain["name"],
            const.chain_type_ids["NONPOLYMER"],
            lig_chain["entity_id"],
            lig_chain["sym_id"],
            lig_chain["asym_id"],
            lig_chain_atom_start,  # atom_idx
            lig_chain_atom_num,  # atom_num
            lig_chain_res_start,  # res_idx
            lig_chain_res_num,  # res_num
            lig_chain["cyclic_period"],
        ))

        # ========================================================================
        # STEP 3: Parse CCD residues and add extra atoms if needed
        # ========================================================================
        parsed_lig = []
        affinity_mw = 0
        
        for local_res_idx, kept_res in enumerate(kept_residues):
            glob_res_idx = kept_res["res_idx"]
            res_name = kept_res["name"]
            
            if res_name not in self._refmol_cache:
                raise ValueError(f"Residue {res_name} not found in reference molecule cache")
            
            ref_mol = get_mol(res_name, self._refmol_cache, self.mol_dir)
            affinity_mw += Descriptors.MolWt(ref_mol)

            parsed = parse_ccd_residue(
                name=res_name,
                ref_mol=ref_mol,
                res_idx=local_res_idx,
                drop_leaving_atoms=True,
            )
            print("parsed:", parsed.name, parsed.idx)

            revised_atoms = []
            for atom in parsed.atoms:
                if (lig_asym_id, glob_res_idx, atom.name) in glob_idx_map:
                    # Atom already exists, get its index and coordinates
                    glob_atom_idx = glob_idx_map[(lig_asym_id, glob_res_idx, atom.name)]
                    if trunk_coords is not None and glob_atom_idx < len(trunk_coords):
                        atom_coord = trunk_coords[glob_atom_idx].copy()
                        # Check if coordinates are all zero (masked), use original if so
                        if np.allclose(atom_coord, 0.0):
                            atom_coord = np.array(atom.coords)
                    else:
                        atom_coord = np.array(atom.coords)
                else:
                    # New atom not in structure, add it
                    glob_atom_idx = global_atom_idx
                    glob_idx_map[(lig_asym_id, glob_res_idx, atom.name)] = glob_atom_idx
                    atom_to_token[glob_atom_idx] = token_idx
                    atom_coord = np.array([0.0, 0.0, 0.0])  # Missing atom coordinates
                    
                    # Add token for extra atom
                    token_data.append((
                        token_idx,  # token_idx
                        glob_atom_idx,  # atom_idx
                        1,  # atom_num
                        local_res_idx,  # res_idx (use local index for MSA compatibility)
                        unk_prot_id,  # res_type
                        res_name,  # res_name
                        lig_chain["sym_id"],  # sym_id
                        lig_chain["asym_id"],  # asym_id
                        lig_chain["entity_id"],  # entity_id
                        const.chain_type_ids["NONPOLYMER"],  # mol_type
                        glob_atom_idx,  # center_idx
                        glob_atom_idx,  # disto_idx
                        atom_coord,  # center_coords
                        atom_coord,  # disto_coords
                        True,  # resolved_mask
                        True,  # disto_mask
                        True,  # modified
                        np.eye(3).flatten(),  # frame_rot
                        np.zeros(3),  # frame_t
                        False,  # frame_mask
                        lig_chain["cyclic_period"],  # cyclic_period
                        True,  # affinity_mask
                    ))
                    
                    atom_data_raw.append((
                        atom.name,
                        #atom.element,#
                        #atom.charge,#
                        atom_coord,
                        #atom_coord, # # conformer same as coords for missing atoms
                        True,  # is_present
                        #atom.chirality, #
                    ))
                    
                    token_idx += 1
                    global_atom_idx += 1

                revised_atoms.append(
                    ParsedAtom(
                        name=atom.name,
                        element=atom.element,
                        charge=atom.charge,
                        coords=atom_coord,
                        conformer=atom.conformer,
                        is_present=atom.is_present,
                        chirality=atom.chirality,
                    )
                )

            parsed_lig.append(
                ParsedResidue(
                    name=parsed.name,
                    type=unk_prot_id,
                    atoms=revised_atoms,
                    bonds=parsed.bonds,
                    idx=parsed.idx, # local res index in kept_residues
                    atom_center=parsed.atom_center,
                    atom_disto=parsed.atom_disto,
                    orig_idx=parsed.orig_idx,
                    is_standard=False, # treat all residues as ligand 
                    is_present=True,
                    rdkit_bounds_constraints=parsed.rdkit_bounds_constraints,
                    chiral_atom_constraints=parsed.chiral_atom_constraints,
                    stereo_bond_constraints=parsed.stereo_bond_constraints,
                    planar_bond_constraints=parsed.planar_bond_constraints,
                    planar_ring_5_constraints=parsed.planar_ring_5_constraints,
                    planar_ring_6_constraints=parsed.planar_ring_6_constraints,
                )
            )
        
        chains = {lig_asym_id: ParsedChain(
            entity=lig_asym_id,
            residues=parsed_lig,
            type=const.chain_type_ids["NONPOLYMER"],
            cyclic_period=0,
            sequence=None,
            affinity=True,
            affinity_mw=affinity_mw,
        )}

        # ========================================================================
        # STEP 4: Build final data structures
        # ========================================================================
        residue_constraints = add_constraints(parsed_lig, lig_asym_id, glob_idx_map, local_to_global_res_idx)
        record = create_metadata(target, chains, lig_asym_id)
        token_bonds, bond_data = add_bonds(parsed_lig, lig_asym_id, glob_idx_map, atom_to_token, local_to_global_res_idx)
        
        # Convert atom_data_raw to AtomV2 format: (name, coords, is_present, bfactor, plddt)
        atom_data = [
            (a[0], a[1], a[2], 0.0, 1.0)  # name, coords, is_present, bfactor, plddt
            for a in atom_data_raw
        ]
        
        connections = []  # FIXME: omit connection constraints in yaml
        bond_data = bond_data + connections
        coords = np.array([(a[1],) for a in atom_data], dtype=Coords)  # Extract coords from atom_data
        mask = np.array([a[2] for a in atom_data], dtype=bool)  # Extract is_present from atom_data
        
        struct_data = StructureV2(
            atoms=np.array(atom_data, dtype=AtomV2),
            bonds=np.array(bond_data, dtype=BondV2),
            residues=np.array(res_data, dtype=Residue),
            chains=np.array(chain_data, dtype=Chain),
            interfaces=np.array([], dtype=Interface),
            mask=mask,
            coords=coords,
            ensemble=np.array([(0, len(coords))], dtype=Ensemble),
        )

        return (
            np.array(token_data, dtype=TokenV2),
            np.array(token_bonds, dtype=TokenBondV2),
            struct_data,
            residue_constraints,
            atom_to_token,
            record
        )

def add_bonds(parsed_res: list[ParsedResidue], lig_asym_id, glob_idx_map: dict, atom_to_token: dict, local_to_global_res_idx: dict) -> tuple[list, list]:
    bond_data=[] #
    token_bonds = []
    for res in parsed_res:
        local_res_idx = res.idx
        glob_res_idx = local_to_global_res_idx[local_res_idx]
        
        # Map bonds: parsed atom indices -> global atom indices -> token indices
        for bond in res.bonds:
            parsed_atom_1 = res.atoms[bond.atom_1].name
            parsed_atom_2 = res.atoms[bond.atom_2].name
            key1 = (lig_asym_id, glob_res_idx, parsed_atom_1)
            key2 = (lig_asym_id, glob_res_idx, parsed_atom_2)
            # Get global atom indices
            if key1 in glob_idx_map and key2 in glob_idx_map:
                global_atom_1 = glob_idx_map[key1]
                global_atom_2 = glob_idx_map[key2]
                bond_data.append(
                    (
                        lig_asym_id,
                        lig_asym_id,
                        local_res_idx,  # Use local index for bond_data
                        local_res_idx,  # Use local index for bond_data
                        global_atom_1,
                        global_atom_2,
                        bond.type,
                    )
                )
                # Get token indices
                if global_atom_1 in atom_to_token and global_atom_2 in atom_to_token:
                    token_1 = atom_to_token[global_atom_1]
                    token_2 = atom_to_token[global_atom_2]
                    token_bonds.append((token_1, token_2, int(bond.type)))
                else:
                    print(f"Warning: Global atom indices {global_atom_1} or {global_atom_2} not found in atom_to_token")
            else:
                print(f"Warning: Parsed atom indices {parsed_atom_1} or {parsed_atom_2} not found in parsed_atom_idx_to_global for residue {local_res_idx} (global: {glob_res_idx})")

    return token_bonds, bond_data

def add_constraints(parsed_res: list[ParsedResidue], lig_asym_id, glob_idx_map: dict, local_to_global_res_idx: dict):
    rdkit_bounds_constraint_data = []
    chiral_atom_constraint_data = []
    stereo_bond_constraint_data = []
    planar_bond_constraint_data = []
    planar_ring_5_constraint_data = []
    planar_ring_6_constraint_data = []

    for res in parsed_res:
        local_res_idx = res.idx
        glob_res_idx = local_to_global_res_idx[local_res_idx]

        if res.rdkit_bounds_constraints is not None:
            for constraint in res.rdkit_bounds_constraints:
                rdkit_bounds_constraint_data.append(  # noqa: PERF401
                    (
                        tuple(
                            glob_idx_map[(lig_asym_id, glob_res_idx, res.atoms[c_atom_idx].name)]
                            for c_atom_idx in constraint.atom_idxs
                        ),
                        constraint.is_bond,
                        constraint.is_angle,
                        constraint.upper_bound,
                        constraint.lower_bound,
                    )
                )
        if res.chiral_atom_constraints is not None:
            for constraint in res.chiral_atom_constraints:
                chiral_atom_constraint_data.append(  # noqa: PERF401
                    (
                        tuple(
                            glob_idx_map[(lig_asym_id, glob_res_idx, res.atoms[c_atom_idx].name)]
                            for c_atom_idx in constraint.atom_idxs
                        ),
                        constraint.is_reference,
                        constraint.is_r,
                    )
                )
        if res.stereo_bond_constraints is not None:
            for constraint in res.stereo_bond_constraints:
                stereo_bond_constraint_data.append(  # noqa: PERF401
                    (
                        tuple(
                            glob_idx_map[(lig_asym_id, glob_res_idx, res.atoms[c_atom_idx].name)]
                            for c_atom_idx in constraint.atom_idxs
                        ),
                        constraint.is_check,
                        constraint.is_e,
                    )
                )
        if res.planar_bond_constraints is not None:
            for constraint in res.planar_bond_constraints:
                planar_bond_constraint_data.append(  # noqa: PERF401
                    (
                        tuple(
                            glob_idx_map[(lig_asym_id, glob_res_idx, res.atoms[c_atom_idx].name)]
                            for c_atom_idx in constraint.atom_idxs
                        ),
                    )
                )
        if res.planar_ring_5_constraints is not None:
            for constraint in res.planar_ring_5_constraints:
                planar_ring_5_constraint_data.append(  # noqa: PERF401
                    (
                        tuple(
                            glob_idx_map[(lig_asym_id, glob_res_idx, res.atoms[c_atom_idx].name)]
                            for c_atom_idx in constraint.atom_idxs
                        ),
                    )
                )
        if res.planar_ring_6_constraints is not None:
            for constraint in res.planar_ring_6_constraints:
                planar_ring_6_constraint_data.append(  # noqa: PERF401
                    (
                        tuple(
                            glob_idx_map[(lig_asym_id, glob_res_idx, res.atoms[c_atom_idx].name)]
                            for c_atom_idx in constraint.atom_idxs
                        ),
                    )
                )

    rdkit_bounds_constraints = np.array(
        rdkit_bounds_constraint_data, dtype=RDKitBoundsConstraint
    )
    chiral_atom_constraints = np.array(
        chiral_atom_constraint_data, dtype=ChiralAtomConstraint
    )
    stereo_bond_constraints = np.array(
        stereo_bond_constraint_data, dtype=StereoBondConstraint
    )
    planar_bond_constraints = np.array(
        planar_bond_constraint_data, dtype=PlanarBondConstraint
    )
    planar_ring_5_constraints = np.array(
        planar_ring_5_constraint_data, dtype=PlanarRing5Constraint
    )
    planar_ring_6_constraints = np.array(
        planar_ring_6_constraint_data, dtype=PlanarRing6Constraint
    )

    return ResidueConstraints(
        rdkit_bounds_constraints=rdkit_bounds_constraints,
        chiral_atom_constraints=chiral_atom_constraints,
        stereo_bond_constraints=stereo_bond_constraints,
        planar_bond_constraints=planar_bond_constraints,
        planar_ring_5_constraints=planar_ring_5_constraints,
        planar_ring_6_constraints=planar_ring_6_constraints,
    )

def create_metadata(
    target,
    chains,
    lig_asym_id: int=1,
    pocket_constraints=None,
    contact_constraints=None,
    template_records=None,
):
    """
    Create and return metadata objects for a given target structure.

    Parameters:
        target: Target structure object
        chains: Dict of ParsedChain objects
        lig_asym_id: Ligand chain asymmetric ID
        pocket_constraints (optional): Pocket constraints for inference.
        contact_constraints (optional): Contact constraints for inference.
        template_records (optional): List of template records.

    Returns:
        record (Record): Metadata Record object.
    """
    chain_infos = []
    affinity_info = None
    
    # Add receptor chain info (hardcoded to asym_id 0)
    rec_chain_info = [info for info in target.record.chains if info.chain_id == 0]  # FIXME: hardcode rec chain id to 0
    chain_infos.extend(rec_chain_info)

    # Add ligand chain info
    if lig_asym_id in chains:
        chain = chains[lig_asym_id]
        chain_info = ChainInfo(
            chain_id=int(lig_asym_id),
            chain_name=chain.name if hasattr(chain, 'name') else str(lig_asym_id),
            mol_type=int(chain.type),
            cluster_id=-1,
            msa_id=-1,  # empty
            num_residues=len(chain.residues),
            valid=True,
            entity_id=int(chain.entity),
        )
        chain_infos.append(chain_info)
        
        affinity_info = AffinityInfo(
            chain_id=lig_asym_id,
            mw=chain.affinity_mw,
        )

    options = InferenceOptions(
        pocket_constraints=pocket_constraints, 
        contact_constraints=contact_constraints
    )
    record = Record(
        id=target.record.id,
        structure=target.record.structure,
        chains=chain_infos,
        interfaces=[],
        inference_options=options,
        templates=template_records if template_records is not None else [],
        affinity=affinity_info,
    )
    return record
