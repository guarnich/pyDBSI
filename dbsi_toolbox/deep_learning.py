# dbsi_toolbox/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import sys

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DBSI_PhysicsDecoder(nn.Module):
    """
    Physics-Informed Decoder Layer.
    This layer has NO trainable parameters. It strictly implements the DBSI 
    physical equation (Eq. 1 from Wang et al.) to transform predicted tissue 
    parameters back into a diffusion signal.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, n_iso_bases: int = 50):
        super().__init__()
        
        # Register acquisition protocol as fixed buffers (not trainable)
        # bvals: (N_meas,)
        # bvecs: (N_meas, 3)
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))
        
        # Define the fixed isotropic spectrum grid (e.g., 0 to 3.0 um^2/ms)
        # This corresponds to the "L" spectral components in the paper
        iso_grid = torch.linspace(0, 3.0e-3, n_iso_bases)
        self.register_buffer('iso_diffusivities', iso_grid)
        
        self.n_iso = n_iso_bases

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Reconstructs MRI Signal from Microstructural Parameters.
        
        Args:
            params: Tensor of shape (Batch_Size, N_Params) containing:
                    - indices [0 : n_iso] -> Isotropic weights (fractions)
                    - index [n_iso]       -> Fiber fraction
                    - index [n_iso+1]     -> Theta (polar angle)
                    - index [n_iso+2]     -> Phi (azimuthal angle)
                    - index [n_iso+3]     -> Axial Diffusivity (D_ax)
                    - index [n_iso+4]     -> Radial Diffusivity (D_rad)
        
        Returns:
            Signal: Tensor of shape (Batch_Size, N_measurements)
        """
        # 1. Unpack Parameters
        f_iso_weights = params[:, :self.n_iso]        # Isotropic spectral weights
        f_fiber       = params[:, self.n_iso]         # Fiber fraction
        theta         = params[:, self.n_iso + 1]     # Fiber orientation theta
        phi           = params[:, self.n_iso + 2]     # Fiber orientation phi
        d_ax          = params[:, self.n_iso + 3]     # Axial Diffusivity
        d_rad         = params[:, self.n_iso + 4]     # Radial Diffusivity

        # 2. Reconstruct Fiber Direction Vector (Spherical -> Cartesian)
        # Shape: (Batch, 3)
        fiber_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)

        # 3. Compute Anisotropic Component (The Fiber)
        # Calculate angle between fiber direction and gradient vectors
        # bvecs: (N_meas, 3), fiber_dir: (Batch, 3) -> cos_angle: (Batch, N_meas)
        cos_angle = torch.matmul(fiber_dir, self.bvecs.T)
        
        # Apparent Diffusion Coefficient along each gradient direction
        # D_app = D_rad + (D_ax - D_rad) * cos^2(alpha)
        d_app_fiber = d_rad.unsqueeze(1) + (d_ax.unsqueeze(1) - d_rad.unsqueeze(1)) * (cos_angle ** 2)
        
        # Anisotropic Signal Attenuation: exp(-b * D_app)
        signal_fiber = torch.exp(-self.bvals.unsqueeze(0) * d_app_fiber)
        
        # Weighted by fiber fraction
        term_fiber = f_fiber.unsqueeze(1) * signal_fiber

        # 4. Compute Isotropic Spectrum Components
        # bvals: (N_meas), iso_diff: (N_iso) -> basis_matrix: (N_meas, N_iso)
        # Basis function: exp(-b * D_iso)
        basis_iso = torch.exp(-torch.ger(self.bvals, self.iso_diffusivities))
        
        # Sum weighted isotropic components: Matrix multiplication
        # Weights: (Batch, N_iso) @ Basis: (N_iso, N_meas) -> (Batch, N_meas)
        term_iso = torch.matmul(f_iso_weights, basis_iso.T)

        # 5. Combine Components
        # S = f_fiber * S_fiber + sum(f_iso * S_iso)
        signal_pred = term_fiber + term_iso
        
        return signal_pred


class DBSI_Encoder(nn.Module):
    """
    The Neural Network that 'learns' to map DWI signals to DBSI parameters.
    This replaces the iterative solvers (NNLS/NLLS).
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 50):
        super().__init__()
        self.n_iso = n_iso_bases
        
        # Total output parameters: 
        # n_iso (weights) + 1 (fiber_frac) + 2 (angles) + 2 (diffusivities)
        self.n_output = n_iso_bases + 5
        
        self.net = nn.Sequential(
            nn.Linear(n_input_meas, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.n_output)
        )
        
        # Specific activations to enforce physical constraints
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus() # Ensures positivity

    def forward(self, x):
        raw_out = self.net(x)
        
        # Split outputs to apply specific constraints
        # 1. Fractions (Weights & Fiber): Must be positive. 
        # We use Softmax over ALL fractions (iso + fiber) to ensure sum <= 1 conservation
        fractions_raw = raw_out[:, :self.n_iso + 1]
        fractions = torch.softmax(fractions_raw, dim=1) # Sums to 1
        
        f_iso = fractions[:, :self.n_iso]
        f_fiber = fractions[:, self.n_iso]
        
        # 2. Angles (Theta, Phi): 
        # Theta: [0, pi] -> Sigmoid * pi
        theta = self.sigmoid(raw_out[:, self.n_iso + 1]) * np.pi
        # Phi: [-pi, pi] -> (Sigmoid - 0.5) * 2pi
        phi = (self.sigmoid(raw_out[:, self.n_iso + 2]) - 0.5) * 2 * np.pi
        
        # 3. Diffusivities (D_ax, D_rad): Must be positive.
        # We scale them to reasonable physiological ranges (e.g., max 3.0e-3)
        # Sigmoid * 3.0e-3
        d_ax = self.sigmoid(raw_out[:, self.n_iso + 3]) * 3.0e-3
        d_rad = self.sigmoid(raw_out[:, self.n_iso + 4]) * 3.0e-3
        
        # Constraint: D_ax must be >= D_rad
        d_ax = torch.max(d_ax, d_rad)

        # Reassemble parameters
        # [f_iso..., f_fiber, theta, phi, d_ax, d_rad]
        params = torch.cat([
            f_iso, 
            f_fiber.unsqueeze(1), 
            theta.unsqueeze(1), 
            phi.unsqueeze(1), 
            d_ax.unsqueeze(1), 
            d_rad.unsqueeze(1)
        ], dim=1)
        
        return params


class DBSI_DeepSolver:
    """
    The high-level Orchestrator for Self-Supervised Deep Learning DBSI.
    It takes the volume, trains the network specifically on that volume, 
    and returns the parameter maps.
    """
    def __init__(self, n_iso_bases: int = 50, epochs: int = 50, batch_size: int = 4096, learning_rate: float = 0.001):
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fits the DBSI model to the entire volume using Self-Supervised Learning.
        """
        print(f"[DeepSolver] Preparing data for self-supervised training on {DEVICE}...")
        
        # 1. Data Preparation (Flattening)
        # Extract only valid voxels inside the mask to save memory and computation
        X_vol, Y_vol, Z_vol, N_meas = volume.shape
        valid_signals = volume[mask] # Shape: (N_valid_voxels, N_meas)
        
        # Normalize signals (S/S0)
        # Assuming the first volume or low-b volumes are b=0. Simple normalization strategy:
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            valid_signals = valid_signals / (s0 + 1e-6)
        
        # Convert to PyTorch Tensors
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Initialize Networks
        encoder = DBSI_Encoder(n_input_meas=N_meas, n_iso_bases=self.n_iso_bases).to(DEVICE)
        decoder = DBSI_PhysicsDecoder(bvals, bvecs, n_iso_bases=self.n_iso_bases).to(DEVICE)
        optimizer = optim.Adam(encoder.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        # 3. Training Loop (Self-Supervised)
        print(f"[DeepSolver] Training on {len(valid_signals)} voxels for {self.epochs} epochs...")
        encoder.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False, file=sys.stdout):
                x_batch = batch[0].to(DEVICE)
                
                optimizer.zero_grad()
                
                # A. Forward: Encoder predicts parameters
                predicted_params = encoder(x_batch)
                
                # B. Reconstruction: Decoder simulates signal from parameters
                reconstructed_signal = decoder(predicted_params)
                
                # C. Loss: Compare Simulated vs Real Input Signal
                loss = loss_fn(reconstructed_signal, x_batch)
                
                # D. Backward
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Print average loss every few epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: Avg Loss = {epoch_loss / len(dataloader):.6f}")
        
        # 4. Final Inference
        print("[DeepSolver] Running final inference on full volume...")
        encoder.eval()
        
        # Create a dataloader for inference (no shuffle, deterministic order)
        full_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_params_list = []
        
        with torch.no_grad():
            for batch in tqdm(full_loader, desc="Inference", file=sys.stdout):
                x_batch = batch[0].to(DEVICE)
                preds = encoder(x_batch)
                all_params_list.append(preds.cpu().numpy())
        
        # Concatenate all predictions: Shape (N_valid_voxels, N_params)
        flat_predictions = np.concatenate(all_params_list, axis=0)
        
        # 5. Volume Reconstruction (1D -> 3D Mapping)
        print("[DeepSolver] Reconstructing 3D maps...")
        
        # Parse predictions back into maps
        # Indices map:
        # 0 to n_iso-1  -> Isotropic Weights
        # n_iso         -> Fiber Fraction
        # n_iso+1       -> Theta
        # n_iso+2       -> Phi
        # n_iso+3       -> D_ax
        # n_iso+4       -> D_rad
        
        n_iso = self.n_iso_bases
        
        # Helper to map 1D array back to 3D volume
        def map_to_3d(flat_data):
            vol_3d = np.zeros((X_vol, Y_vol, Z_vol), dtype=np.float32)
            vol_3d[mask] = flat_data
            return vol_3d
            
        # Calculate aggregated isotropic fractions
        iso_weights = flat_predictions[:, :n_iso]
        iso_grid_np = np.linspace(0, 3.0e-3, n_iso)
        
        # Thresholds (Wang et al.)
        mask_res = iso_grid_np <= 0.3e-3
        mask_hin = (iso_grid_np > 0.3e-3) & (iso_grid_np <= 2.0e-3)
        mask_wat = iso_grid_np > 2.0e-3
        
        f_restricted_flat = np.sum(iso_weights[:, mask_res], axis=1)
        f_hindered_flat   = np.sum(iso_weights[:, mask_hin], axis=1)
        f_water_flat      = np.sum(iso_weights[:, mask_wat], axis=1)
        
        # Fiber properties
        f_fiber_flat = flat_predictions[:, n_iso]
        d_ax_flat    = flat_predictions[:, n_iso + 3]
        d_rad_flat   = flat_predictions[:, n_iso + 4]
        
        # Angles to Vector (X, Y, Z)
        theta = flat_predictions[:, n_iso + 1]
        phi   = flat_predictions[:, n_iso + 2]
        dir_x = np.sin(theta) * np.cos(phi)
        dir_y = np.sin(theta) * np.sin(phi)
        dir_z = np.cos(theta)
        
        # Build Output Dictionary
        output_maps = {
            'restricted_fraction': map_to_3d(f_restricted_flat),
            'hindered_fraction':   map_to_3d(f_hindered_flat),
            'water_fraction':      map_to_3d(f_water_flat),
            'fiber_fraction':      map_to_3d(f_fiber_flat),
            'axial_diffusivity':   map_to_3d(d_ax_flat),
            'radial_diffusivity':  map_to_3d(d_rad_flat),
            'fiber_dir_x':         map_to_3d(dir_x),
            'fiber_dir_y':         map_to_3d(dir_y),
            'fiber_dir_z':         map_to_3d(dir_z),
        }
        
        print("[DeepSolver] Done.")
        return output_maps