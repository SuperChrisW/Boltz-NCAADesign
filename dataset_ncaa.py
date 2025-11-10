from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader

from boltz.data.module.inferencev2 import PredictionDataset, load_input, collate
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.mol import load_molecules
from boltz.data.types import Manifest
from boltz.data import const
from .tokenizer_ncaa import NCAA_Tokenizer

class NCAA_PredictionDataset(PredictionDataset):
    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Path,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
        target_res_idx: list = [0],
        tokenize_res_window: int = 0,
        max_tokens: int = 256,
        max_atoms: int = 2048,
        trunk_coords: np.ndarray = None,
    ):
        super().__init__(
            manifest=manifest,
            target_dir=target_dir,
            msa_dir=msa_dir,
            mol_dir=mol_dir,
            constraints_dir=constraints_dir,
            template_dir=template_dir,
            extra_mols_dir=extra_mols_dir,
            override_method=override_method,
            affinity=affinity,
        )
        self.tokenizer = NCAA_Tokenizer(mol_dir=self.mol_dir)
        self.cropper = AffinityCropper()
        self.target_res_idx = target_res_idx if affinity else None
        self.tokenize_res_window = tokenize_res_window if affinity else None
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.trunk_coords = trunk_coords

    def __getitem__(self, idx: int):
        record = self.manifest.records[idx]
        input_data = load_input(
            record=record,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            constraints_dir=self.constraints_dir, # protein-protein complex prediction has empty constraints, need add extra constraints for ligand
            template_dir=self.template_dir,
            extra_mols_dir=self.extra_mols_dir,
            affinity=True,
        ) # normal protein-peptide input_data
        
        assert len(self.target_res_idx) > idx
        # extract k-mer residues from peptide chain and tokenize as ligand
        tokenized, contact_constraints = self.tokenizer.NCAA_tokenize(input_data, res_id=self.target_res_idx[idx], tokenize_res_window=self.tokenize_res_window,
                                                trunk_coords=self.trunk_coords)
        tokenized = self.cropper.crop(
            tokenized, max_tokens=self.max_tokens, max_atoms=self.max_atoms
        )

        molecules = {**self.canonicals, **input_data.extra_mols}
        mol_names = set(tokenized.tokens["res_name"].tolist()) - set(molecules.keys())
        if mol_names:
            molecules.update(load_molecules(self.mol_dir, mol_names))

        # TODO: revise the pocket_constraints for the extracted ligand
        options = record.inference_options
        pocket_constraints = getattr(options, "pocket_constraints", None) if options else None
        # contact_constraints are now generated from trunk_coords in tokenizer

        features = self.featurizer.process(
            tokenized,
            molecules=molecules,
            random=np.random.default_rng(42),
            training=False,
            max_atoms=None,
            max_tokens=None,
            max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            inference_pocket_constraints=pocket_constraints,
            inference_contact_constraints=contact_constraints if contact_constraints else None,
            compute_constraint_features=True,
            override_method=self.override_method,
            compute_affinity=self.affinity,
        )

        features["record"] = record

        # Save features for future comparison
        import os
        import pickle
        save_features_path = self.target_dir / "features" / f"{record.id}_test.pkl"
        os.makedirs(os.path.dirname(save_features_path), exist_ok=True)
        save_data = dict(features)  # Make a shallow copy
        # Features might include non-serializable data, so filter them if needed
        try:
            with open(save_features_path, "ab") as f:
                pickle.dump(save_data, f)
        except Exception as save_exc:
            print(f"Warning: Could not save features for {record.id}: {save_exc}")
        return features

def make_ncaa_loader(dataset: NCAA_PredictionDataset, batch_size: int = 1, num_workers: int = 2) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=collate)

# from Boltz2InferenceDataModule.transfer_batch_to_device()
def transfer_batch_to_device(
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "affinity_mw",
            ]:
                batch[key] = batch[key].to(device)
        return batch
