from __future__ import annotations
import pickle
import warnings
from pathlib import Path
from typing import Dict, List
from rdkit import Chem

from boltz.data.parse.yaml import parse_yaml
from boltz.data.mol import load_canonicals
from boltz.data.types import Manifest, Record
from boltz.data import const

def load_ccd(mol_dir: Path, boltz2: bool):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Depickling from a version number.*")
        return load_canonicals(str(mol_dir)) if boltz2 else None

def prepare_inputs(*, data_yaml: Path, ccd, mol_dir: Path,
                   msa_dir: Path, processed_msa_dir: Path,
                   constraints_dir: Path, templates_dir: Path,
                   mols_out_dir: Path, structure_dir: Path, records_dir: Path,
                   logger) -> Manifest:
    """Inline of your process_input (no logic change)."""
    target = parse_yaml(Path(data_yaml), ccd, str(mol_dir), True)
    target_id = target.record.id
    prot_id = const.chain_type_ids["PROTEIN"]

    # Assign MSA ids for protein chains
    to_generate = {}
    for chain in target.record.chains:
        if (chain.mol_type == prot_id) and (chain.msa_id == 0):
            entity_id = chain.entity_id
            msa_id = f"{target_id}_{entity_id}"
            to_generate[msa_id] = target.sequences[entity_id]
            chain.msa_id = msa_dir / f"{msa_id}.csv"
            logger.info(f"Chain {chain.id}: MSA ID {chain.msa_id}")
        elif chain.msa_id == 0:
            chain.msa_id = -1

    # Map unique MSA ids
    msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
    msa_id_map: Dict[Path, str] = {}
    for msa_idx, msa_id in enumerate(msas):
        msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

    # Dump templates
    for template_id, template in target.templates.items():
        template_path = templates_dir / f"{target.record.id}_{template_id}.npz"
        template.dump(template_path)

    # Dump constraints
    (constraints_dir / f"{target.record.id}.npz").parent.mkdir(parents=True, exist_ok=True)
    target.residue_constraints.dump(constraints_dir / f"{target.record.id}.npz")

    # Dump extra molecules
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with (mols_out_dir / f"{target.record.id}.pkl").open("wb") as f:
        pickle.dump(target.extra_mols, f)

    # Dump structure
    target.structure.dump(structure_dir / f"{target.record.id}.npz")

    # Dump record
    target.record.dump(records_dir / f"{target.record.id}.json")

    records: List[Record] = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(records_dir.parent / "manifest.json")
    return manifest
