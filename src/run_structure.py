from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from .forward import trunk_forward
from .io_utils import write_predictions, set_record_affinity

def run_structure_once(*, model, data_module: Boltz2InferenceDataModule,
                       device, recycling_steps, num_sampling_steps,
                       diffusion_samples, max_parallel_samples,
                       structure_dir, logger):
    dl = data_module.predict_dataloader()
    # single-batch run (your original behavior)
    feats = next(iter(dl))
    data_module.transfer_batch_to_device(feats, device, 0)
    model.to(device).eval()

    out = trunk_forward(
        model, feats,
        recycling_steps=recycling_steps,
        num_sampling_steps=num_sampling_steps,
        diffusion_samples=diffusion_samples,
        max_parallel_samples=max_parallel_samples,
        run_confidence_sequentially=True,
    )
    out.update({
        "masks": feats["atom_pad_mask"].expand(out["sample_atom_coords"].shape[0], -1),
        "coords": out["sample_atom_coords"],
        "confidence_score": (
            4 * out["complex_plddt"] + (out["iptm"] if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"])) else out["ptm"])
        ) / 5,
    })
    # mark ligand chain 1 for pre-affinity
    #feats["record"][0] = set_record_affinity(feats["record"][0], chain_id=1)
    write_predictions(out, feats["record"], structure_dir, structure_dir, output_format="pdb", boltz2=True, write_embeddings=False)
    return feats, out
