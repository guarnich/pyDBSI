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

class DBSISyntheticGenerator:
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, d_res_threshold: float = 0.3e-3):
        self.bvals = bvals
        self.bvecs = bvecs
        self.n_meas = len(bvals)
        self.d_res_threshold = d_res_threshold

    def generate_batch(self, batch_size: int, snr: float = 30.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generazione Scenario-Based (Bilanciata)
        n_scenarios = 4
        n_per = batch_size // n_scenarios
        
        # Setup Frazioni per i 4 scenari
        fracs = []
        for _ in range(n_per): fracs.append([np.random.uniform(0.6, 1.0), 0, 0, 0]) # Fiber
        for _ in range(n_per): fracs.append([0, 0, np.random.uniform(0.5, 1.0), 0]) # Hindered
        for _ in range(n_per): fracs.append([0, np.random.uniform(0.4, 0.9), 0, 0]) # Restricted
        for _ in range(n_per): fracs.append([0, 0, 0, np.random.uniform(0.8, 1.0)]) # Water
        
        # Fill remaining samples if batch_size not divisible by 4
        rem_samples = batch_size - (n_per * n_scenarios)
        for _ in range(rem_samples): fracs.append([0.25, 0.25, 0.25, 0.25])

        base_fracs = np.array(fracs)
        noise_fracs = np.random.dirichlet((0.5, 0.5, 0.5, 0.5), len(base_fracs))
        
        # Target order: [fiber, res, hin, wat] -> Mapped to: [f_res, f_hin, f_wat, f_fib]
        f_fib = base_fracs[:, 0] * 0.8 + noise_fracs[:, 0] * 0.2
        f_res = base_fracs[:, 1] * 0.8 + noise_fracs[:, 1] * 0.2
        f_hin = base_fracs[:, 2] * 0.8 + noise_fracs[:, 2] * 0.2
        f_wat = base_fracs[:, 3] * 0.8 + noise_fracs[:, 3] * 0.2
        
        # Normalize
        tot = f_res + f_hin + f_wat + f_fib
        f_res, f_hin, f_wat, f_fib = f_res/tot, f_hin/tot, f_wat/tot, f_fib/tot

        # Parametri Fisici
        N = len(f_fib)
        d_res = np.random.uniform(0.0, self.d_res_threshold, N)
        d_hin = np.random.uniform(self.d_res_threshold + 0.1e-3, 2.5e-3, N)
        # FIX: d_wat scalar is safer for broadcasting
        d_wat = 3.0e-3 
        d_ax = np.random.uniform(1.0e-3, 2.5e-3, N)
        d_rad = np.random.uniform(0.1e-3, 0.6e-3, N)
        
        # Direzioni
        theta = np.arccos(2 * np.random.rand(N) - 1)
        phi = 2 * np.pi * np.random.rand(N)
        fiber_dir = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=1)

        # Forward Model
        # bvecs shape (M, 3), fiber_dir shape (N, 3) -> dot -> (N, M)
        cos_angles = np.dot(fiber_dir, self.bvecs.T)
        d_app_fib = d_rad[:,None] + (d_ax[:,None] - d_rad[:,None]) * cos_angles**2
        
        # Calcolo Segnale (Corretto broadcasting)
        # bvals shape (M,) -> Broadcasting (N, 1) * (1, M) -> (N, M)
        S = (f_res[:,None] * np.exp(-self.bvals * d_res[:,None]) +
             f_hin[:,None] * np.exp(-self.bvals * d_hin[:,None]) +
             f_wat[:,None] * np.exp(-self.bvals * d_wat) + # d_wat è scalare, ok
             f_fib[:,None] * np.exp(-self.bvals * d_app_fib))

        # Rumore Rician
        if snr > 0:
            sigma = 1.0 / snr
            n1 = np.random.normal(0, sigma, S.shape)
            n2 = np.random.normal(0, sigma, S.shape)
            S = np.sqrt((S + n1)**2 + n2**2)

        # Normalize S0
        b0_idx = self.bvals < 50
        # Usa media b0 se esistono, altrimenti usa la prima misura come S0 approx
        if np.sum(b0_idx) > 0:
            b0_mean = np.mean(S[:, b0_idx], axis=1, keepdims=True)
        else:
            b0_mean = S[:, 0:1] # Fallback

        S = S / (b0_mean + 1e-9)
        
        X = torch.tensor(S, dtype=torch.float32)
        Y = torch.tensor(np.stack([f_res, f_hin, f_wat, f_fib], axis=1), dtype=torch.float32)
        
        return X, Y

class DBSI_PriorNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 4),    nn.Softmax(dim=1)
        )
    def forward(self, x): return self.net(x)

class DBSI_DeepRegularizer:
    def __init__(self, bvals, bvecs, epochs=50, batch_size=1024, lr=1e-3, d_res_threshold=0.3e-3):
        self.bvals = bvals
        self.bvecs = bvecs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gen = DBSISyntheticGenerator(bvals, bvecs, d_res_threshold)
        self.model = None

    def train_on_synthetic(self, n_samples=50000, target_snr=30.0):
        print(f"[DL] Generating {n_samples} samples (SNR={target_snr:.1f})...")
        X, Y = self.gen.generate_batch(n_samples, snr=target_snr)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        
        self.model = DBSI_PriorNet(len(self.bvals)).to(DEVICE)
        opt = optim.AdamW(self.model.parameters(), lr=self.lr)
        crit = nn.HuberLoss()
        
        self.model.train()
        for _ in tqdm(range(self.epochs), desc="Training DL", file=sys.stdout):
            epoch_loss = 0
            for bx, by in dl:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                loss = crit(self.model(bx), by)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

    def predict_volume(self, vol, mask):
        self.model.eval()
        X, Y, Z, N = vol.shape
        valid = vol[mask]
        
        # Normalizzazione Reale S0
        b0_idx = self.bvals < 50
        if b0_idx.sum() > 0:
            s0 = np.mean(valid[:, b0_idx], axis=1, keepdims=True)
            valid = valid / (s0 + 1e-6)
        else:
             valid = valid / (valid[:, 0:1] + 1e-6) # Fallback
        
        ds = TensorDataset(torch.tensor(valid, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=4096)
        preds = []
        with torch.no_grad():
            for bx in dl:
                preds.append(self.model(bx[0].to(DEVICE)).cpu().numpy())
        
        flat = np.concatenate(preds, 0)
        maps = {}
        for i, k in enumerate(['dl_restricted_fraction', 'dl_hindered_fraction', 'dl_water_fraction', 'dl_fiber_fraction']):
            m = np.zeros((X,Y,Z))
            m[mask] = flat[:, i]
            maps[k] = m
        return maps