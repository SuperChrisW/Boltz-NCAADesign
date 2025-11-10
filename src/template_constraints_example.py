"""
Example usage of the template_constraints module.

This example shows how to:
1. Load reference coordinates from .npz files
2. Create and initialize the ligand template module
3. Use it during structure prediction to bias the diffusion sampling
"""

from __future__ import annotations
from pathlib import Path
import torch
from .template_constraints import LigandTemplateModule, LigandTemplateLoader
from .model_loader import create_ligand_template_module, load_template_coords
from .forward import trunk_forward


def example_usage_with_npz_file():
    """
    Example: Using template constraints with a pre-stored .npz file.
    
    The .npz file should contain:
    - 'protein_coords': numpy array of shape [N_prot_atoms, 3] with protein reference coordinates
    - 'ligand_coords': numpy array of shape [N_lig_atoms, 3] with ligand reference coordinates
    - Optional: 'protein_mask' and 'ligand_mask' for masking invalid atoms
    """
    
    # 1. Load reference coordinates from .npz file
    template_path = Path("path/to/reference_coords.npz")
    reference_coords = load_template_coords(
        template_path=template_path,
        protein_key="protein_coords",  # Key in .npz file
        ligand_key="ligand_coords",     # Key in .npz file
    )
    
    # 2. Create the ligand template module
    # token_z should match your model's token_z dimension (typically 128 for Boltz2)
    ligand_template_module = create_ligand_template_module(
        token_z=128,
        template_dim=128,
        template_blocks=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    # 3. Use during forward pass
    # Assuming you have a model and feats
    # model = ...  # Your Boltz2 model
    # feats = ...  # Your input features
    
    # trunk_forward will now incorporate template constraints
    # out = trunk_forward(
    #     model,
    #     feats,
    #     recycling_steps=5,
    #     num_sampling_steps=200,
    #     diffusion_samples=1,
    #     max_parallel_samples=1,
    #     ligand_template_module=ligand_template_module,
    #     reference_coords=reference_coords,
    # )


def example_using_template_loader():
    """
    Example: Using the LigandTemplateLoader for multiple templates.
    """
    
    # Initialize loader
    loader = LigandTemplateLoader(
        template_dir=Path("path/to/templates/"),
        protein_key="protein_coords",
        ligand_key="ligand_coords",
    )
    
    # Load multiple templates (cached automatically)
    template1 = loader.load_template("template1.npz")
    template2 = loader.load_template("template2.npz")
    
    # Clear cache if needed
    loader.clear_cache()


def example_npz_file_format():
    """
    Example of the expected .npz file format.
    
    You can create such a file using numpy:
    
    import numpy as np
    
    # Reference coordinates
    protein_coords = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # [N_prot, 3]
    ligand_coords = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # [N_lig, 3]
    
    # Optional masks (1 = valid, 0 = invalid)
    protein_mask = np.ones(N_prot, dtype=bool)  # All atoms valid
    ligand_mask = np.ones(N_lig, dtype=bool)    # All atoms valid
    
    # Save to .npz file
    np.savez(
        "reference_coords.npz",
        protein_coords=protein_coords,
        ligand_coords=ligand_coords,
        protein_mask=protein_mask,
        ligand_mask=ligand_mask,
    )
    """
    pass


def example_integration_with_run_affinity():
    """
    Example: Integrating template constraints into the affinity evaluation pipeline.
    
    This shows how to modify run_affinity.py to use template constraints.
    """
    
    # In your main script:
    # 1. Load template coordinates
    template_path = Path("path/to/reference_coords.npz")
    reference_coords = load_template_coords(template_path)
    
    # 2. Create template module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ligand_template_module = create_ligand_template_module(
        token_z=128,  # Match your model's token_z
        device=device,
    )
    
    # 3. Modify trunk_forward call in run_affinity.py:
    # out_trunk = trunk_forward(
    #     model_struct,
    #     batch_ncaa,
    #     recycling_steps=cfg.predict_args_affinity["recycling_steps"],
    #     num_sampling_steps=cfg.predict_args_affinity["sampling_steps"],
    #     diffusion_samples=cfg.predict_args_affinity["diffusion_samples"],
    #     max_parallel_samples=cfg.predict_args_affinity["max_parallel_samples"],
    #     run_confidence_sequentially=True,
    #     ligand_template_module=ligand_template_module,  # Add this
    #     reference_coords=reference_coords,              # Add this
    # )


if __name__ == "__main__":
    print("Template constraints module usage examples")
    print("See function docstrings for implementation details")

