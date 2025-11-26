# dbsi_toolbox/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
from typing import Dict, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. GENERATORE SINTETICO "SCENARIO-BASED"
# ==========================================
class DBSISyntheticGenerator:
    """
    Genera dati sintetici bilanciati per coprire i casi clinici reali.
    Invece di frazioni casuali uniformi, simula voxel specifici:
    1. Sostanza Bianca Sana (Alta Fibra)
    2. Edema/Danno Tissutale (Alto Hindered)
    3. Infiammazione/Tumore (Alto Restricted)
    4. CSF/Cisti (Alta Acqua Libera)
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray):
        self.bvals = bvals
        self.bvecs = bvecs
        self.n_meas = len(bvals)

    def generate_batch(self, batch_size: int, snr: float = 30.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dividiamo il batch in 4 scenari
        n_scenarios = 4
        n_per_scen = batch_size // n_scenarios
        
        # --- SCENARIO 1: Fiber Dominant (White Matter) ---
        # Fiber > 60%, il resto rumore isotropo
        f_fib_1 = np.random.uniform(0.60, 0.95, n_per_scen)
        rem_1 = 1.0 - f_fib_1
        # Distribuisci il rimanente casualmente
        iso_mix_1 = np.random.dirichlet((1, 1, 1), n_per_scen) 
        f_res_1 = rem_1 * iso_mix_1[:, 0]
        f_hin_1 = rem_1 * iso_mix_1[:, 1]
        f_wat_1 = rem_1 * iso_mix_1[:, 2]

        # --- SCENARIO 2: Hindered Dominant (Edema/Grey Matter) ---
        # Hindered > 50%, Fiber bassa
        f_hin_2 = np.random.uniform(0.50, 0.90, n_per_scen)
        rem_2 = 1.0 - f_hin_2
        iso_mix_2 = np.random.dirichlet((1, 2, 1), n_per_scen) # [res, fib, wat]
        f_res_2 = rem_2 * iso_mix_2[:, 0]
        f_fib_2 = rem_2 * iso_mix_2[:, 1]
        f_wat_2 = rem_2 * iso_mix_2[:, 2]

        # --- SCENARIO 3: Restricted Dominant (High Cellularity) ---
        # Restricted > 40% (Raro ma critico da imparare)
        f_res_3 = np.random.uniform(0.40, 0.80, n_per_scen)
        rem_3 = 1.0 - f_res_3
        iso_mix_3 = np.random.dirichlet((1, 1, 1), n_per_scen)
        f_hin_3 = rem_3 * iso_mix_3[:, 0]
        f_fib_3 = rem_3 * iso_mix_3[:, 1]
        f_wat_3 = rem_3 * iso_mix_3[:, 2]

        # --- SCENARIO 4: Water Dominant (CSF/Necrosis) ---
        # Water > 80%
        f_wat_4 = np.random.uniform(0.80, 1.00, n_per_scen)
        rem_4 = 1.0 - f_wat_4
        iso_mix_4 = np.random.dirichlet((1, 1, 1), n_per_scen)
        f_res_4 = rem_4 * iso_mix_4[:, 0]
        f_hin_4 = rem_4 * iso_mix_4[:, 1]
        f_fib_4 = rem_4 * iso_mix_4[:, 2]

        # Concatenazione
        f_fiber = np.concatenate([f_fib_1, f_fib_2, f_fib_3, f_fib_4])
        f_res = np.concatenate([f_res_1, f_res_2, f_res_3, f_res_4])
        f_hin = np.concatenate([f_hin_1, f_hin_2, f_hin_3, f_hin_4])
        f_wat = np.concatenate([f_wat_1, f_wat_2, f_wat_3, f_wat_4])

        # Shuffle per non polarizzare il batch
        perm = np.random.permutation(len(f_fiber))
        f_fiber, f_res, f_hin, f_wat = f_fiber[perm], f_res[perm], f_hin[perm], f_wat[perm]

        # --- Parametri Fisici (Strict Bounds) ---
        batch_real_size = len(f_fiber)
        
        # Restricted: [0, 0.2] um2/ms (Strictly < 0.3)
        d_res = np.random.uniform(0.0, 0.2e-3, batch_real_size)
        
        # Hindered: [0.8, 2.0] um2/ms (Separation gap from Restricted)
        d_hin = np.random.uniform(0.8e-3, 2.0e-3, batch_real_size)
        
        # Water: 3.0 fix
        d_wat = 3.0e-3
        
        # Fiber: [1.0, 2.5] axial, [0.1, 0.6] radial
        d_ax = np.random.uniform(1.0e-3, 2.5e-3, batch_real_size)
        d_rad = np.random.uniform(0.1e-3, 0.6e-3, batch_real_size)

        # Fiber Orientation
        theta = np.arccos(2 * np.random.rand(batch_real_size) - 1)
        phi = 2 * np.pi * np.random.rand(batch_real_size)
        fiber_dir = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=1)

        # --- Forward Model ---
        cos_angles = np.dot(fiber_dir, self.bvecs.T)
        d_app_fiber = d_rad[:, None] + (d_ax[:, None] - d_rad[:, None]) * (cos_angles**2)
        
        sig_fib = np.exp(-self.bvals[None, :] * d_app_fiber)
        sig_res = np.exp(-self.bvals[None, :] * d_res[:, None])
        sig_hin = np.exp(-self.bvals[None, :] * d_hin[:, None])
        sig_wat = np.exp(-self.bvals[None, :] * d_wat)
        
        clean_signal = (
            f_fiber[:, None] * sig_fib +
            f_res[:, None] * sig_res +
            f_hin[:, None] * sig_hin +
            f_wat[:, None] * sig_wat
        )

        # Rumore Rician
        sigma = 1.0 / snr
        noise_real = np.random.normal(0, sigma, clean_signal.shape)
        noise_imag = np.random.normal(0, sigma, clean_signal.shape)
        noisy_signal = np.sqrt((clean_signal + noise_real)**2 + noise_imag**2)
        
        # Normalizzazione S0
        b0_mean = np.mean(noisy_signal[:, self.bvals < 50], axis=1, keepdims=True) + 1e-9
        noisy_signal = noisy_signal / b0_mean

        # Tensori per PyTorch
        X = torch.tensor(noisy_signal, dtype=torch.float32)
        # Labels Order: [Restricted, Hindered, Water, Fiber]
        Y = torch.tensor(np.stack([f_res, f_hin, f_wat, f_fiber], axis=1), dtype=torch.float32)

        return X, Y

# ==========================================
# 2. RETE NEURALE
# ==========================================
class DBSI_PriorNet(nn.Module):
    def __init__(self, n_input_meas: int):
        super().__init__()
        # Rete leggermente più profonda per catturare le non-linearità difficili
        self.net = nn.Sequential(
            nn.Linear(n_input_meas, 512),
            nn.BatchNorm1d(512),
            nn.GELU(), # GELU performa meglio di ReLU per regressione fisica
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, 4), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. REGOLARIZZATORE
# ==========================================
class DBSI_DeepRegularizer:
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, epochs: int = 50, 
                 batch_size: int = 1024, lr: float = 1e-3):
        self.bvals = bvals
        self.bvecs = bvecs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.generator = DBSISyntheticGenerator(bvals, bvecs)

    def train_on_synthetic(self, n_samples=100000, target_snr=30.0):
        # Aumentiamo i sample a 100k per coprire bene tutti gli scenari
        print(f"[DL-Reg] Generating {n_samples} Scenario-Based samples (SNR={target_snr:.1f})...")
        
        X, Y = self.generator.generate_batch(n_samples, snr=target_snr)
        
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = DBSI_PriorNet(n_input_meas=len(self.bvals)).to(DEVICE)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss() # Più robusto agli outlier del MSE
        
        self.model.train()
        print(f"[DL-Reg] Training Prior Network on {DEVICE}...")
        
        pbar = tqdm(range(self.epochs), desc="Training Priors", file=sys.stdout)
        for _ in pbar:
            epoch_loss = 0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / len(loader)})

    def predict_volume(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None: raise RuntimeError("Model not trained.")
        
        self.model.eval()
        X_dim, Y_dim, Z_dim, N_meas = volume.shape
        valid_signals = volume[mask]
        
        # Normalizzazione
        b0_idx = self.bvals < 50
        if np.sum(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            valid_signals = valid_signals / (s0 + 1e-9)
            
        # Batch Inference
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        preds_list = []
        with torch.no_grad():
            for batch in loader:
                bx = batch[0].to(DEVICE)
                preds_list.append(self.model(bx).cpu().numpy())
                
        flat_preds = np.concatenate(preds_list, axis=0)
        
        maps = {}
        # Order: [Restricted, Hindered, Water, Fiber]
        keys = ['dl_restricted_fraction', 'dl_hindered_fraction', 'dl_water_fraction', 'dl_fiber_fraction']
        
        for i, k in enumerate(keys):
            vol = np.zeros((X_dim, Y_dim, Z_dim))
            vol[mask] = flat_preds[:, i]
            maps[k] = vol
            
        return maps