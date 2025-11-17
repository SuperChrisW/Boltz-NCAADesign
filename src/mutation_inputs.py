from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import yaml

AA_3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "SEC": "U", "PYL": "O",
}


@dataclass
class MutationJob:
    pdb_entry: str
    mutation: str
    residue_index: int
    data_yaml: Path
    config_yaml: Path


def parse_mutation_label(label: str) -> Tuple[str, str, int, str]:
    """
    Parse a mutation string such as 'LI45G' → (orig='L', chain='I', pdb_idx=45, mutated='G').
    """
    pattern = r"^([A-Z])([A-Za-z])(\d+)([A-Z])$"
    match = re.match(pattern, label.strip())
    if not match:
        raise ValueError(f"Invalid mutation label '{label}'")
    orig, chain_id, resnum, mutated = match.groups()
    return orig, chain_id.upper(), int(resnum), mutated


def extract_chain_sequences(pdb_file: Path) -> Dict[str, "OrderedDict[int, str]"]:
    """
    Extract ordered residue maps (PDB residue number -> three-letter code) per chain.
    """
    chains: Dict[str, "OrderedDict[int, str]"] = {}
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    with pdb_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21:22].strip()
            resnum_str = line[22:26].strip()
            resname = line[17:20].strip().upper()
            atom_name = line[12:16].strip()
            if not chain_id or atom_name != "CA":
                continue
            try:
                resnum = int(resnum_str)
            except ValueError:
                continue
            if chain_id not in chains:
                chains[chain_id] = OrderedDict()
            # Store first occurrence only
            chains[chain_id].setdefault(resnum, resname)
    return chains


def build_sequence(chain_residues: "OrderedDict[int, str]") -> Tuple[str, List[int]]:
    """
    Convert a residue mapping into a 1-letter sequence and parallel residue-number list.
    """
    letters: List[str] = []
    residue_numbers: List[int] = []
    for resnum, resname in chain_residues.items():
        one_letter = AA_3_TO_1.get(resname, "X")
        letters.append(one_letter)
        residue_numbers.append(resnum)
    return "".join(letters), residue_numbers


def mutate_sequence(sequence: str, residue_numbers: List[int], target_resnum: int,
                    original: str, mutated: str) -> Tuple[str, int]:
    """
    Apply a point mutation at the residue with PDB numbering target_resnum.

    Returns mutated sequence string and the 1-indexed sequence position mutated.
    """
    if target_resnum not in residue_numbers:
        raise ValueError(f"Residue {target_resnum} not present in chain (available: {residue_numbers[:5]} ...)")
    seq_idx = residue_numbers.index(target_resnum)
    seq_chars = list(sequence)
    observed = seq_chars[seq_idx]
    if observed != original:
        raise ValueError(f"Original residue mismatch: expected {original}, found {observed} at index {target_resnum}")
    seq_chars[seq_idx] = mutated
    return "".join(seq_chars), seq_idx + 1


def write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

def generate_job(
        jobs: List[MutationJob],
        pdb_entry: str,
        sub_mutation: str,
        chain_ids: List[str],
        chain_sequences: Dict[str, "OrderedDict[int, str]"],
        output_root: Path,
        orig: str,
        chain_id: str,
        resnum: int,
        mutated: str,
        tokenize_window: int,
    ) -> None:
        seq, res_numbers = build_sequence(chain_sequences[chain_id])
        mutated_seq, seq_idx = mutate_sequence(seq, res_numbers, resnum, orig, mutated)

        partner_chain = [cid for cid in chain_ids if cid != chain_id]
        if len(partner_chain) != 1:
            raise ValueError(f"Partner chain resolution failed for {pdb_entry}")
        partner_id = partner_chain[0]
        if partner_id not in chain_sequences:
            raise ValueError(f"Partner chain {partner_id} missing in PDB {pdb_id}")
        partner_seq, _ = build_sequence(chain_sequences[partner_id])

        yaml_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": partner_id, "sequence": partner_seq, "msa": "empty"}},
                {"protein": {"id": chain_id, "sequence": mutated_seq, "msa": "empty"}},
            ],
        }

        job_dir = output_root / pdb_entry / sub_mutation
        data_yaml = job_dir / "complex.yaml"
        config_yaml = job_dir / "run_config.yaml"
        out_dir = job_dir / "outputs"

        write_yaml(yaml_data, data_yaml)
        config_data = {
            "data_yaml": str(data_yaml),
            "out_dir": str(out_dir),
            "residue_min": seq_idx,
            "residue_max": seq_idx,
            "tokenize_res_window": tokenize_window,
        }
        write_yaml(config_data, config_yaml)

        jobs.append(MutationJob(
            pdb_entry=pdb_entry,
            mutation=sub_mutation,
            residue_index=seq_idx,
            data_yaml=data_yaml,
            config_yaml=config_yaml,
        ))
        return jobs

def generate_inputs_from_mutations(csv_path: Path, pdb_dir: Path, output_root: Path,
                                   tokenize_window: int = 0) -> List[MutationJob]:
    """
    Parse mutation_validation_results.csv and emit per-mutation YAML + config overrides.
    """
    csv_path = Path(csv_path).expanduser()
    pdb_dir = Path(pdb_dir).expanduser()
    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs: List[MutationJob] = []
    pdb_cache: Dict[str, Dict[str, "OrderedDict[int, str]"]] = {}

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_entry = (row.get("pdb_entry") or row.get("#Pdb") or "").strip()
            mutation_label = (row.get("mutation") or row.get("Mutation(s)_PDB") or "").strip()
            valid_flag = str(row.get("valid", "")).strip().lower() == "true"
            if not pdb_entry or not mutation_label or not valid_flag:
                continue

            parts = pdb_entry.split("_")
            if len(parts) < 3:
                raise ValueError(f"Unexpected pdb_entry format '{pdb_entry}'")
            pdb_id = parts[0]
            chain_ids = parts[1:]
            if len(chain_ids) != 2:
                raise ValueError(f"Expected exactly 2 chains in pdb_entry '{pdb_entry}'")

            if pdb_id not in pdb_cache:
                pdb_cache[pdb_id] = extract_chain_sequences(pdb_dir / f"{pdb_id}.pdb")

                # Generate job for wildtype sequence at the mutation site
                for sub_mutation in mutation_label.split("_"):
                    orig, chain_id, resnum, mutated = parse_mutation_label(sub_mutation)
                    if chain_id not in chain_ids:
                        raise ValueError(f"Mutation chain {chain_id} not found in {pdb_entry}")
                    if chain_id not in pdb_cache[pdb_id]:
                        raise ValueError(f"Chain {chain_id} not extracted from PDB {pdb_id}")

                    generate_job(
                        jobs=jobs,
                        pdb_entry=pdb_entry,
                        sub_mutation=f"{orig}{chain_id}{resnum}{orig}",  # wildtype, not mutated
                        chain_ids=chain_ids,
                        chain_sequences=pdb_cache[pdb_id],
                        output_root=output_root,
                        orig=orig,
                        chain_id=chain_id,
                        resnum=resnum,
                        mutated=orig,
                        tokenize_window=tokenize_window,
                    )
            chain_sequences = pdb_cache[pdb_id]

            for sub_mutation in mutation_label.split("_"):
                orig, chain_id, resnum, mutated = parse_mutation_label(sub_mutation)
                if chain_id not in chain_ids:
                    raise ValueError(f"Mutation chain {chain_id} not found in {pdb_entry}")
                if chain_id not in chain_sequences:
                    raise ValueError(f"Chain {chain_id} not extracted from PDB {pdb_id}")

                generate_job(
                    jobs=jobs,
                    pdb_entry=pdb_entry,
                    sub_mutation=sub_mutation,
                    chain_ids=chain_ids,
                    chain_sequences=chain_sequences,
                    output_root=output_root,
                    orig=orig,
                    chain_id=chain_id,
                    resnum=resnum,
                    mutated=mutated,
                    tokenize_window=tokenize_window,
                )
    return jobs


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Boltz inputs from mutation_validation_results.csv")
    parser.add_argument("--csv", type=Path, required=True, help="Path to mutation_validation_results.csv")
    parser.add_argument("--pdb-dir", type=Path, required=True, help="Directory containing source PDB files")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for generated job folders")
    parser.add_argument("--tokenize-window", type=int, default=0, help="tokenize_res_window override per job")
    args = parser.parse_args(argv)

    jobs = generate_inputs_from_mutations(
        csv_path=args.csv,
        pdb_dir=args.pdb_dir,
        output_root=args.output_root,
        tokenize_window=args.tokenize_window,
    )
    print(f"Generated {len(jobs)} jobs in {args.output_root}")
    for job in jobs[:3]:
        print(f"- {job.pdb_entry} {job.mutation}: cfg={job.config_yaml} yaml={job.data_yaml}")


if __name__ == "__main__":
    main()
