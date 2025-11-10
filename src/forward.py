from __future__ import annotations
import torch
from typing import Optional
from .template_constraints import LigandTemplateModule

def trunk_forward(model, feats, *, recycling_steps: int, num_sampling_steps: int,
                  diffusion_samples: int, max_parallel_samples: int,
                  run_confidence_sequentially: bool = True,
                  ligand_template_module: Optional[LigandTemplateModule] = None,
                  reference_coords: Optional[dict] = None) -> dict:
    """Structure trunk + diffusion + confidence (logic unchanged)."""
    with torch.no_grad():
        s_inputs = model.input_embedder(feats)
        s_init = model.s_init(s_inputs)
        z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]

        rpe = model.rel_pos(feats)
        z_init = z_init + rpe + model.token_bonds(feats["token_bonds"].float())
        if model.bond_type_feature:
            z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + model.contact_conditioning(feats)

        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        if model.run_trunk_and_structure:
            for i in range(recycling_steps + 1):
                s = s_init + model.s_recycle(model.s_norm(s))
                z = z_init + model.z_recycle(model.z_norm(z))

                if model.use_templates:
                    tm = model.template_module._orig_mod if model.is_template_compiled and not model.training else model.template_module
                    z = z + tm(z, feats, pair_mask, use_kernels=model.use_kernels)

                # Add ligand template constraints if provided
                if ligand_template_module is not None:
                    z = z + ligand_template_module(
                        z, feats, pair_mask,
                        reference_coords=reference_coords,
                        use_kernels=model.use_kernels if hasattr(model, 'use_kernels') else False,
                    )

                mm = model.msa_module._orig_mod if model.is_msa_compiled and not model.training else model.msa_module
                z = z + mm(z, s_inputs, feats, use_kernels=model.use_kernels)

                pm = model.pairformer_module._orig_mod if model.is_pairformer_compiled and not model.training else model.pairformer_module
                s, z = pm(s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels)

        pdist = model.distogram_module(z)
        out = {"pdistogram": pdist, "s": s, "z": z}

        # Diffusion cond
        q, c, to_keys, aeb, adb, ttb = model.diffusion_conditioning(s, z, rpe, feats)
        cond = {"q": q, "c": c, "to_keys": to_keys, "atom_enc_bias": aeb, "atom_dec_bias": adb, "token_trans_bias": ttb}

        # Sample
        struct_out = model.structure_module.sample(
            s_trunk=s.float(), s_inputs=s_inputs.float(), feats=feats,
            num_sampling_steps=num_sampling_steps, atom_mask=feats["atom_pad_mask"].float(),
            multiplicity=diffusion_samples, max_parallel_samples=max_parallel_samples,
            steering_args=model.steering_args, diffusion_conditioning=cond,
        )
        out.update(struct_out)

        # Confidence
        out.update(
            model.confidence_module(
                s_inputs=s_inputs, s=s, z=z,
                x_pred=(out["sample_atom_coords"] if not model.skip_run_structure else feats["coords"].repeat_interleave(diffusion_samples, 0)),
                feats=feats,
                pred_distogram_logits=out["pdistogram"][:, :, :, 0],
                multiplicity=diffusion_samples,
                run_sequentially=run_confidence_sequentially,
                use_kernels=model.use_kernels,
            )
        )
        return out

def affinity_forward(model_aff, feats, trunk_out: dict) -> dict:
    """Affinity heads (unchanged) using trunk z and chosen coords."""
    z = trunk_out["z"]
    pad_token_mask = feats["token_pad_mask"][0]
    rec_mask = (feats["mol_type"][0] == 0) * pad_token_mask
    lig_mask = feats["affinity_token_mask"][0].to(torch.bool) * pad_token_mask
    cross_pair_mask = (
        lig_mask[:, None] * rec_mask[None, :]
        + rec_mask[:, None] * lig_mask[None, :]
        + lig_mask[:, None] * lig_mask[None, :]
    )
    z_aff = z * cross_pair_mask[None, :, :, None]

    best_idx = torch.argsort(trunk_out["iptm"], descending=True)[0].item()
    coords_aff = trunk_out["sample_atom_coords"].detach()[0][None, None] # FIXME: use 0-th coords 

    s_inputs = model_aff.input_embedder(feats, affinity=True)

    with torch.no_grad():
        out1 = model_aff.affinity_module1(s_inputs=s_inputs, z=z_aff, x_pred=coords_aff, feats=feats, multiplicity=1, use_kernels=model_aff.use_kernels)
        out1["affinity_probability_binary"] = torch.sigmoid(out1["affinity_logits_binary"])
        out2 = model_aff.affinity_module2(s_inputs=s_inputs, z=z_aff, x_pred=coords_aff, feats=feats, multiplicity=1, use_kernels=model_aff.use_kernels)
        out2["affinity_probability_binary"] = torch.sigmoid(out2["affinity_logits_binary"])

        trunk_out.update({
            "affinity_pred_value": (out1["affinity_pred_value"] + out2["affinity_pred_value"]) / 2,
            "affinity_probability_binary": (out1["affinity_probability_binary"] + out2["affinity_probability_binary"]) / 2,
            "affinity_pred_value1": out1["affinity_pred_value"],
            "affinity_probability_binary1": out1["affinity_probability_binary"],
            "affinity_pred_value2": out2["affinity_pred_value"],
            "affinity_probability_binary2": out2["affinity_probability_binary"],
        })
    return trunk_out
