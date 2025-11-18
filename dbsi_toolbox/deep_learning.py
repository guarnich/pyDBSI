# dbsi_toolbox/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
from typing import Dict

# Configurazione Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DBSI_PhysicsDecoder(nn.Module):
    """
    Decodificatore Fisico (Non addestrabile).
    Implementa l'equazione DBSI esatta (Wang et al., 2011).
    Agisce come funzione di perdita "fisica": trasforma i parametri stimati
    in un segnale MRI sintetico da confrontare con quello reale.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, n_iso_bases: int = 50):
        super().__init__()
        
        # Buffer fissi per la geometria di acquisizione
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))
        
        # Griglia spettrale fissa [0, 3.0] um^2/ms
        iso_grid = torch.linspace(0, 3.0e-3, n_iso_bases)
        self.register_buffer('iso_diffusivities', iso_grid)
        
        self.n_iso = n_iso_bases

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # Unpacking dei parametri predetti dall'encoder
        # params: [f_iso (N), f_fiber (1), theta, phi, D_ax, D_rad]
        f_iso_weights = params[:, :self.n_iso]
        f_fiber       = params[:, self.n_iso]
        theta         = params[:, self.n_iso + 1]
        phi           = params[:, self.n_iso + 2]
        d_ax          = params[:, self.n_iso + 3]
        d_rad         = params[:, self.n_iso + 4]

        # 1. Componente Anisotropa (Fibre)
        # Conversione angoli sferici -> Vettore Cartesiano
        fiber_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)

        # Calcolo D_apparente per ogni direzione di gradiente
        # cos_alpha = dot(fiber_dir, gradient_dir)
        cos_angle = torch.matmul(fiber_dir, self.bvecs.T)
        d_app_fiber = d_rad.unsqueeze(1) + (d_ax.unsqueeze(1) - d_rad.unsqueeze(1)) * (cos_angle ** 2)
        
        # Segnale fibra: f_fib * exp(-b * D_app)
        signal_fiber = f_fiber.unsqueeze(1) * torch.exp(-self.bvals.unsqueeze(0) * d_app_fiber)

        # 2. Componente Isotropa (Spettro)
        # Matrice base isotropa: exp(-b * D_iso)
        basis_iso = torch.exp(-torch.ger(self.bvals, self.iso_diffusivities))
        
        # Segnale isotropo: somma pesata delle basi
        signal_iso = torch.matmul(f_iso_weights, basis_iso.T)

        # 3. Segnale Totale
        return signal_fiber + signal_iso


class DBSI_RegularizedMLP(nn.Module):
    """
    Encoder MLP con Regolarizzazione Architetturale.
    Mappa il segnale DWI grezzo -> Parametri DBSI fisici.
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 50, dropout_rate: float = 0.1):
        super().__init__()
        self.n_iso = n_iso_bases
        
        # Neuroni totali di output
        self.n_output = n_iso_bases + 5 
        
        # Architettura "Bottleneck" per forzare la compressione dell'informazione
        # e filtrare il rumore.
        self.net = nn.Sequential(
            nn.Linear(n_input_meas, 128),
            nn.LayerNorm(128),           # Normalizzazione per stabilità
            nn.ELU(),
            nn.Dropout(dropout_rate),    # Regolarizzazione stocastica
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, self.n_output)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raw = self.net(x)
        
        # --- Constraints Fisici (Hard Regularization) ---
        
        # 1. Frazioni (devono sommare a 1)
        # Usiamo Softmax su TUTTE le frazioni (isotrope + fibra)
        fractions_raw = raw[:, :self.n_iso + 1]
        fractions = torch.softmax(fractions_raw, dim=1)
        
        f_iso = fractions[:, :self.n_iso]
        f_fiber = fractions[:, self.n_iso]
        
        # 2. Angoli (Limitati al range geometrico)
        theta = self.sigmoid(raw[:, self.n_iso + 1]) * np.pi        # [0, pi]
        phi   = (self.sigmoid(raw[:, self.n_iso + 2]) - 0.5) * 2 * np.pi # [-pi, pi]
        
        # 3. Diffusività (Limitate a range fisiologici umani)
        # Evita che il fit "esploda" verso valori non fisici per fittare il rumore
        d_ax = self.sigmoid(raw[:, self.n_iso + 3]) * 3.0e-3    # Max 3.0 um2/ms
        d_rad = self.sigmoid(raw[:, self.n_iso + 4]) * 3.0e-3
        
        # Constraint aggiuntivo: D_ax >= D_rad (definizione di fibra)
        d_ax = torch.max(d_ax, d_rad + 1e-6) # +epsilon per stabilità numerica

        return torch.cat([
            f_iso, 
            f_fiber.unsqueeze(1), 
            theta.unsqueeze(1), 
            phi.unsqueeze(1), 
            d_ax.unsqueeze(1), 
            d_rad.unsqueeze(1)
        ], dim=1)


class DBSI_DeepSolver:
    """
    Solver Self-Supervised con Denoising Regularization.
    Impara dai dati del paziente specifico.
    """
    def __init__(self, 
                 n_iso_bases: int = 50, 
                 epochs: int = 100, 
                 batch_size: int = 2048, 
                 learning_rate: float = 1e-3,
                 noise_injection_level: float = 0.02): # 2% noise injection
        
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.noise_level = noise_injection_level
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        
        print(f"[DeepSolver] Avvio training self-supervised su dispositivo: {DEVICE}")
        
        # 1. Preparazione Dati
        X_vol, Y_vol, Z_vol, N_meas = volume.shape
        valid_signals = volume[mask]
        
        # Normalizzazione S0 robusta
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            # Evita divisione per zero e sopprimi background
            s0[s0 < 1e-3] = 1.0 
            valid_signals = valid_signals / s0
            # Clip per rimuovere outlier estremi prima del training
            valid_signals = np.clip(valid_signals, 0, 1.5)
        
        # Dataset PyTorch
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Inizializzazione Modello
        encoder = DBSI_RegularizedMLP(
            n_input_meas=N_meas, 
            n_iso_bases=self.n_iso_bases,
            dropout_rate=0.1
        ).to(DEVICE)
        
        decoder = DBSI_PhysicsDecoder(
            bvals, 
            bvecs, 
            n_iso_bases=self.n_iso_bases
        ).to(DEVICE)
        
        # Optimizer con Weight Decay (L2 Regularization)
        optimizer = optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Loss Function: L1 Loss è più robusta agli outlier della MSE
        loss_fn = nn.L1Loss()
        
        # 3. Training Loop con Denoising
        print(f"[DeepSolver] Training su {len(valid_signals)} voxel (Epochs={self.epochs})...")
        encoder.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch in tqdm(dataloader, desc=f"Epoca {epoch+1}", leave=False, file=sys.stdout):
                clean_signal = batch[0].to(DEVICE)
                
                # --- TECNICA: Denoising Autoencoder ---
                # Aggiungiamo rumore all'input, ma calcoliamo la loss sul segnale pulito.
                # Questo forza la rete a imparare la struttura sottostante e ignorare il rumore.
                noise = torch.randn_like(clean_signal) * self.noise_level
                noisy_input = clean_signal + noise
                
                optimizer.zero_grad()
                
                # A. Predizione Parametri (da input rumoroso)
                params_pred = encoder(noisy_input)
                
                # B. Ricostruzione Fisica (segnale teorico)
                signal_recon = decoder(params_pred)
                
                # C. Loss (confronto con segnale originale del paziente)
                loss = loss_fn(signal_recon, clean_signal)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoca {epoch+1}: Loss = {epoch_loss / len(dataloader):.6f}")
        
        # 4. Inferenza Finale (senza dropout/noise)
        print("[DeepSolver] Generazione mappe volumetriche...")
        encoder.eval()
        full_loader = DataLoader(dataset, batch_size=self.batch_size*2, shuffle=False)
        all_preds = []
        
        with torch.no_grad():
            for batch in full_loader:
                x = batch[0].to(DEVICE)
                preds = encoder(x)
                all_preds.append(preds.cpu().numpy())
                
        flat_results = np.concatenate(all_preds, axis=0)
        
        # 5. Ricostruzione 3D e Aggregazione Metriche
        print("[DeepSolver] Aggregazione metriche DBSI...")
        n_iso = self.n_iso_bases
        
        # Helper map 3D
        def to_3d(flat_arr):
            vol = np.zeros((X_vol, Y_vol, Z_vol), dtype=np.float32)
            vol[mask] = flat_arr
            return vol
            
        # Estrazione componenti
        iso_weights = flat_results[:, :n_iso]
        fiber_frac  = flat_results[:, n_iso]
        theta       = flat_results[:, n_iso+1]
        phi         = flat_results[:, n_iso+2]
        d_ax        = flat_results[:, n_iso+3]
        d_rad       = flat_results[:, n_iso+4]
        
        # Aggregazione Spettrale (Restricted / Hindered / Free)
        # La griglia è lineare da 0 a 3.0
        grid = np.linspace(0, 3.0e-3, n_iso)
        mask_res = grid <= 0.3e-3
        mask_hin = (grid > 0.3e-3) & (grid <= 2.0e-3)
        mask_wat = grid > 2.0e-3
        
        f_res = np.sum(iso_weights[:, mask_res], axis=1)
        f_hin = np.sum(iso_weights[:, mask_hin], axis=1)
        f_wat = np.sum(iso_weights[:, mask_wat], axis=1)
        
        # Conversione Direzioni
        dir_x = np.sin(theta) * np.cos(phi)
        dir_y = np.sin(theta) * np.sin(phi)
        dir_z = np.cos(theta)
        
        return {
            'restricted_fraction': to_3d(f_res),
            'hindered_fraction':   to_3d(f_hin),
            'water_fraction':      to_3d(f_wat),
            'fiber_fraction':      to_3d(fiber_frac),
            'axial_diffusivity':   to_3d(d_ax),
            'radial_diffusivity':  to_3d(d_rad),
            'fiber_dir_x':         to_3d(dir_x),
            'fiber_dir_y':         to_3d(dir_y),
            'fiber_dir_z':         to_3d(dir_z),
        }