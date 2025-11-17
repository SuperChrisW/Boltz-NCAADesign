"""
Template constraints module for incorporating protein-ligand position reference.

This module loads pre-stored .npz coordinate files containing reference positions
for protein-ligand complexes and incorporates these features into the trunk to bias
the diffusion-based structure prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple
from torch import nn, Tensor
from torch.nn.functional import one_hot

from boltz.data import const
from boltz.model.layers.pairformer import PairformerNoSeqModule


class LigandTemplateModule(nn.Module):
    """
    Module for incorporating ligand position reference as template constraints.
    
    Similar to TemplateModule but designed for ligand-protein interactions.
    Processes reference coordinates from .npz files and adds features to
    pairwise embeddings to bias the diffusion sampling.
    """

    def __init__(
        self,
        token_z: int=128,
        template_dim: int = 64,
        template_blocks: int = 2,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        use_kernels: bool = False,
    ) -> None:
        """
        Initialize the ligand template module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.
        template_dim : int
            Dimension for template feature processing.
        template_blocks : int
            Number of pairformer blocks for template processing.
        dropout : float
            Dropout rate.
        pairwise_head_width : int
            Width of pairwise attention heads.
        pairwise_num_heads : int
            Number of pairwise attention heads.
        post_layer_norm : bool
            Whether to use post layer normalization.
        activation_checkpointing : bool
            Whether to use activation checkpointing.
        min_dist : float
            Minimum distance for distogram bins (Angstroms).
        max_dist : float
            Maximum distance for distogram bins (Angstroms).
        num_bins : int
            Number of distance bins.
        use_kernels : bool
            Whether to use optimized kernels.
        """
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.use_kernels = use_kernels
        self.relu = nn.ReLU()
        
        # Normalization layers
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        
        # Projections
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        
        # Feature projection: distogram (num_bins) + mask (1) + unit_vector (3) + frame_mask (1)
        # = num_bins + 5 features per pair
        self.a_proj = nn.Linear(num_bins + 5, template_dim, bias=False)
        
        # Output projection
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        
        # Pairformer for processing template features
        self.pairformer = PairformerNoSeqModule(
            template_dim,
            num_blocks=template_blocks,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
            activation_checkpointing=activation_checkpointing,
        )

    def load_reference_coords(
        self,
        npz_path: Path,
        protein_key: str = "protein_coords",
        ligand_key: str = "ligand_coords",
        protein_mask_key: Optional[str] = None,
        ligand_mask_key: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load reference coordinates from .npz file.

        Parameters
        ----------
        npz_path : Path
            Path to the .npz file containing reference coordinates.
        protein_key : str
            Key for protein coordinates in .npz file.
        ligand_key : str
            Key for ligand coordinates in .npz file.
        protein_mask_key : str, optional
            Key for protein mask in .npz file.
        ligand_mask_key : str, optional
            Key for ligand mask in .npz file.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing loaded coordinates and masks.
        """
        if not npz_path.exists():
            raise FileNotFoundError(f"Template coordinates file not found: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        
        result = {}
        
        # Load protein coordinates
        if protein_key in data:
            result["protein_coords"] = data[protein_key]
        else:
            # Try alternative keys
            for alt_key in ["rec_coords", "receptor_coords", "coords"]:
                if alt_key in data:
                    result["protein_coords"] = data[alt_key]
                    break
            if "protein_coords" not in result:
                raise KeyError(f"Protein coordinates not found. Tried: {protein_key}, rec_coords, receptor_coords, coords")
        
        # Load ligand coordinates
        if ligand_key in data:
            result["ligand_coords"] = data[ligand_key]
        else:
            # Try alternative keys
            for alt_key in ["lig_coords", "binder_coords"]:
                if alt_key in data:
                    result["ligand_coords"] = data[alt_key]
                    break
            if "ligand_coords" not in result:
                raise KeyError(f"Ligand coordinates not found. Tried: {ligand_key}, lig_coords, binder_coords")
        
        # Load masks if available
        if protein_mask_key and protein_mask_key in data:
            result["protein_mask"] = data[protein_mask_key]
        elif "protein_mask" in data:
            result["protein_mask"] = data["protein_mask"]
        else:
            # Create default mask (all True)
            result["protein_mask"] = np.ones(result["protein_coords"].shape[0], dtype=bool)
        
        if ligand_mask_key and ligand_mask_key in data:
            result["ligand_mask"] = data[ligand_mask_key]
        elif "ligand_mask" in data:
            result["ligand_mask"] = data["ligand_mask"]
        else:
            # Create default mask (all True)
            result["ligand_mask"] = np.ones(result["ligand_coords"].shape[0], dtype=bool)
        
        return result

    def compute_template_features(
        self,
        protein_coords: Tensor,
        ligand_coords: Tensor,
        protein_mask: Optional[Tensor] = None,
        ligand_mask: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Compute template features from reference coordinates.

        Parameters
        ----------
        protein_coords : Tensor
            Reference protein coordinates [N_prot, 3].
        ligand_coords : Tensor
            Reference ligand coordinates [N_lig, 3].
        protein_mask : Tensor, optional
            Mask for protein atoms [N_prot].
        ligand_mask : Tensor, optional
            Mask for ligand atoms [N_lig].
        device : torch.device, optional
            Device to compute on.

        Returns
        -------
        Tensor
            Template features [N_prot, N_lig, template_dim].
        """
        if device is None:
            device = protein_coords.device
        
        N_prot = protein_coords.shape[0]
        N_lig = ligand_coords.shape[0]
        
        # Default masks
        if protein_mask is None:
            protein_mask = torch.ones(N_prot, device=device, dtype=torch.bool)
        if ligand_mask is None:
            ligand_mask = torch.ones(N_lig, device=device, dtype=torch.bool)
        
        # Expand masks to pairwise
        pair_mask = protein_mask[:, None] * ligand_mask[None, :]  # [N_prot, N_lig]
        pair_mask = pair_mask[..., None]  # [N_prot, N_lig, 1]
        
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute pairwise distances
            dists = torch.cdist(protein_coords, ligand_coords)  # [N_prot, N_lig]
            
            # Create distogram
            boundaries = torch.linspace(
                self.min_dist, self.max_dist, self.num_bins - 1,
                device=device, dtype=dists.dtype
            )
            distogram = (dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=self.num_bins)  # [N_prot, N_lig, num_bins]
            
            # Compute unit vectors (direction from protein to ligand)
            # For each protein-ligand pair, compute normalized direction vector
            protein_expanded = protein_coords[:, None, :]  # [N_prot, 1, 3]
            ligand_expanded = ligand_coords[None, :, :]  # [1, N_lig, 3]
            vectors = ligand_expanded - protein_expanded  # [N_prot, N_lig, 3]
            norms = torch.norm(vectors, dim=-1, keepdim=True)
            unit_vectors = torch.where(
                norms > 1e-6,
                vectors / norms,
                torch.zeros_like(vectors)
            )  # [N_prot, N_lig, 3]
            
            # Concatenate features: distogram + mask + unit_vector + frame_mask (placeholder)
            # frame_mask is set to 1.0 for all pairs (assuming all frames are valid)
            frame_mask = torch.ones(N_prot, N_lig, 1, device=device, dtype=unit_vectors.dtype)
            
            a_tij = torch.cat([
                distogram,  # [N_prot, N_lig, num_bins]
                pair_mask.float(),  # [N_prot, N_lig, 1]
                unit_vectors,  # [N_prot, N_lig, 3]
                frame_mask,  # [N_prot, N_lig, 1]
            ], dim=-1)  # [N_prot, N_lig, num_bins + 5]
            
            # Project to template_dim
            a_tij = self.a_proj(a_tij)  # [N_prot, N_lig, template_dim]
        
        return a_tij

    def forward(
        self,
        z: Tensor,
        feats: Dict[str, Tensor],
        pair_mask: Tensor,
        reference_coords: Optional[Dict[str, np.ndarray]] = None,
        use_kernels: bool = False,
    ) -> Tensor:
        """
        Forward pass: incorporate ligand template features into pairwise embeddings.

        Parameters
        ----------
        z : Tensor
            Pairwise embeddings [B, N, N, token_z].
        feats : Dict[str, Tensor]
            Input features dictionary. Should contain:
            - 'asym_id': Chain identifiers [B, N]
            - 'mol_type': Molecule type identifiers [B, N]
            - 'ligand_template_coords': Optional ligand reference coordinates
            - 'protein_template_coords': Optional protein reference coordinates
        pair_mask : Tensor
            Pair mask [B, N, N].
        reference_coords : Dict[str, np.ndarray], optional
            Pre-loaded reference coordinates. If None, will try to extract from feats.
        use_kernels : bool
            Whether to use optimized kernels.

        Returns
        -------
        Tensor
            Updated pairwise embeddings [B, N, N, token_z].
        """
        B, N = z.shape[0], z.shape[1]
        
        # Extract reference coordinates
        if reference_coords is None:
            # Try to get from feats
            if "ligand_template_coords" in feats and "protein_template_coords" in feats:
                protein_coords = feats["protein_template_coords"]
                ligand_coords = feats["ligand_template_coords"]
                protein_mask = feats.get("protein_template_mask", None)
                ligand_mask = feats.get("ligand_template_mask", None)
            else:
                # No template coordinates available, return z unchanged
                return torch.zeros_like(z)
        else:
            # Convert numpy arrays to tensors
            protein_coords = torch.from_numpy(reference_coords["protein_coords"]).to(z.device)
            ligand_coords = torch.from_numpy(reference_coords["ligand_coords"]).to(z.device)
            protein_mask = (
                torch.from_numpy(reference_coords["protein_mask"]).to(z.device)
                if "protein_mask" in reference_coords
                else None
            )
            ligand_mask = (
                torch.from_numpy(reference_coords["ligand_mask"]).to(z.device)
                if "ligand_mask" in reference_coords
                else None
            )
        
        # Identify protein and ligand tokens from feats
        asym_id = feats["asym_id"]  # [B, N]
        mol_type = feats["mol_type"]  # [B, N]
        
        # Assume protein is mol_type == 0 (PROTEIN) and ligand is mol_type == 3 (NONPOLYMER)
        # This may need adjustment based on your specific use case
        protein_token_mask = (mol_type == const.chain_type_ids["PROTEIN"]).float()  # [B, N]
        ligand_token_mask = (mol_type == const.chain_type_ids["NONPOLYMER"]).float()  # [B, N]
        
        # For simplicity, process batch 0 only (can be extended for batched processing)
        # Map tokens to reference coordinates
        # This is a simplified mapping - in practice, you may need more sophisticated
        # alignment between token indices and coordinate indices
        
        # Compute template features for protein-ligand pairs
        # We need to map token indices to coordinate indices
        # For now, assume 1:1 mapping (this should be customized based on your data structure)
        
        # Get number of protein and ligand tokens
        n_prot_tokens = int(protein_token_mask[0].sum().item())
        n_lig_tokens = int(ligand_token_mask[0].sum().item())
        
        if n_prot_tokens == 0 or n_lig_tokens == 0:
            return torch.zeros_like(z)
        
        # Map coordinates to tokens (simplified - assumes sequential mapping)
        # In practice, you may need to use atom_to_token mapping from feats
        n_prot_coords = min(protein_coords.shape[0], n_prot_tokens)
        n_lig_coords = min(ligand_coords.shape[0], n_lig_tokens)
        
        protein_coords_mapped = protein_coords[:n_prot_coords].to(z.device)
        ligand_coords_mapped = ligand_coords[:n_lig_coords].to(z.device)
        
        # Map masks if available
        protein_mask_mapped = None
        ligand_mask_mapped = None
        if protein_mask is not None:
            protein_mask_mapped = protein_mask[:n_prot_coords].to(z.device)
        if ligand_mask is not None:
            ligand_mask_mapped = ligand_mask[:n_lig_coords].to(z.device)
        
        # Compute template features
        template_features = self.compute_template_features(
            protein_coords_mapped,
            ligand_coords_mapped,
            protein_mask=protein_mask_mapped,
            ligand_mask=ligand_mask_mapped,
            device=z.device,
        )  # [n_prot_coords, n_lig_coords, template_dim]
        
        # Project z to template_dim and add template features
        # We need to map template features to the full token space
        z_proj = self.z_proj(self.z_norm(z))  # [B, N, N, template_dim]
        
        # Create a mask for protein-ligand pairs
        prot_mask_expanded = protein_token_mask[:, :, None]  # [B, N, 1]
        lig_mask_expanded = ligand_token_mask[:, None, :]  # [B, 1, N]
        prot_lig_mask = (prot_mask_expanded * lig_mask_expanded)[..., None]  # [B, N, N, 1]
        
        # Map template features to token space
        # For simplicity, we'll use the first n_prot_coords protein tokens
        # and first n_lig_coords ligand tokens
        template_features_expanded = torch.zeros(
            B, N, N, template_features.shape[-1],
            device=z.device, dtype=template_features.dtype
        )
        
        # Place template features in the appropriate positions
        # This is a simplified mapping - should be customized based on actual token-to-coord mapping
        if n_prot_coords > 0 and n_lig_coords > 0:
            # Find protein and ligand token indices
            prot_token_indices = torch.where(protein_token_mask[0] > 0)[0][:n_prot_coords]
            lig_token_indices = torch.where(ligand_token_mask[0] > 0)[0][:n_lig_coords]
            
            # Map template features to token pairs
            for i, p_idx in enumerate(prot_token_indices):
                for j, l_idx in enumerate(lig_token_indices):
                    template_features_expanded[0, p_idx, l_idx, :] = template_features[i, j, :]
                    # Also set reverse (ligand-protein) if symmetric
                    template_features_expanded[0, l_idx, p_idx, :] = template_features[i, j, :]
        
        # Add template features to projected z
        v = z_proj + template_features_expanded * prot_lig_mask  # [B, N, N, template_dim]
        
        # Process through pairformer (expects [B, N, N, D] and [B, N, N])
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
        v = self.v_norm(v)
        
        # Project back to token_z space
        u = self.u_proj(self.relu(v))  # [B, N, N, token_z]
        
        # Only apply to protein-ligand pairs
        u = u * prot_lig_mask
        
        return u


class LigandTemplateLoader:
    """
    Utility class for loading and managing ligand template coordinates.
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        protein_key: str = "protein_coords",
        ligand_key: str = "ligand_coords",
    ):
        """
        Initialize the template loader.

        Parameters
        ----------
        template_dir : Path, optional
            Directory containing template .npz files.
        protein_key : str
            Key for protein coordinates in .npz files.
        ligand_key : str
            Key for ligand coordinates in .npz files.
        """
        self.template_dir = template_dir
        self.protein_key = protein_key
        self.ligand_key = ligand_key
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def load_template(
        self,
        template_path: Path,
        cache: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Load template coordinates from file.

        Parameters
        ----------
        template_path : Path
            Path to the .npz template file.
        cache : bool
            Whether to cache loaded templates.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing loaded coordinates and masks.
        """
        template_path = Path(template_path)
        if not template_path.is_absolute() and self.template_dir:
            template_path = self.template_dir / template_path
        
        cache_key = str(template_path)
        if cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        module = LigandTemplateModule(token_z=128)  # Dummy module for loading
        result = module.load_reference_coords(
            template_path,
            protein_key=self.protein_key,
            ligand_key=self.ligand_key,
        )
        
        if cache:
            self._cache[cache_key] = result
        
        return result

    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()

