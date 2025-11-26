import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DBSI_PhysicsLayer(nn.Module):
    """
    Questo strato non ha pesi da addestrare. Implementa l'equazione DBSI.
    Trasforma i parametri (Frazioni, Diffusività, Angoli) in Segnale DWI.
    """
    def __init__(self, bvals, bvecs):
        super().__init__()
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))

    def forward(self, params):
        # params: [batch, 10] -> [f_res, d_res, f_hin, d_hin, f_wat, f_fib, d_ax, d_rad, theta, phi]
        
        # Estrai parametri (usando sigmoid/softplus per vincolarli fisicamente)
        f_raw = params[:, [0, 2, 4, 5]] # res, hin, wat, fib
        f_norm = torch.softmax(f_raw, dim=1) # Somma a 1
        
        f_res, f_hin, f_wat, f_fib = f_norm[:,0], f_norm[:,1], f_norm[:,2], f_norm[:,3]
        
        # Vincoli fisici "soft" (per aiutare la convergenza)
        d_res = torch.sigmoid(params[:, 1]) * 0.3e-3          # Max 0.3
        d_hin = torch.sigmoid(params[:, 3]) * 2.0e-3 + 0.3e-3 # 0.3 - 2.3
        d_wat = torch.tensor(3.0e-3, device=DEVICE)           # Fisso
        
        d_ax = torch.sigmoid(params[:, 6]) * 3.0e-3           # 0 - 3.0
        d_rad = torch.sigmoid(params[:, 7]) * d_ax            # < d_ax (vincolo anisotropia)
        
        theta = params[:, 8]
        phi = params[:, 9]

        # Calcolo Direzione Fibra
        fib_dir = torch.stack([
            torch.sin(theta)*torch.cos(phi),
            torch.sin(theta)*torch.sin(phi),
            torch.cos(theta)
        ], dim=1)

        # Forward Model (Broadcasting avanzato)
        # bvecs: (N_meas, 3)
        # fib_dir: (Batch, 3)
        cos_angles = torch.matmul(fib_dir, self.bvecs.T) # (Batch, N_meas)
        
        d_app_fib = d_rad[:,None] + (d_ax[:,None] - d_rad[:,None]) * (cos_angles**2)
        
        # Ricostruzione Segnale
        S = (f_res[:,None] * torch.exp(-self.bvals * d_res[:,None]) +
             f_hin[:,None] * torch.exp(-self.bvals * d_hin[:,None]) +
             f_wat[:,None] * torch.exp(-self.bvals * d_wat) +
             f_fib[:,None] * torch.exp(-self.bvals * d_app_fib))
             
        return S, f_norm

class DBSI_Autoencoder(nn.Module):
    def __init__(self, n_meas, bvals, bvecs):
        super().__init__()
        
        # ENCODER: Segnale -> Parametri
        self.encoder = nn.Sequential(
            nn.Linear(n_meas, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 10) # 10 Parametri DBSI latenti
        )
        
        # DECODER: Parametri -> Segnale (Fisica fissa)
        self.decoder = DBSI_PhysicsLayer(bvals, bvecs)

    def forward(self, x):
        params = self.encoder(x)
        signal_pred, fractions = self.decoder(params)
        return signal_pred, fractions, params

class SelfSupervisedSolver:
    def __init__(self, bvals, bvecs, lr=1e-3):
        self.bvals = bvals
        self.bvecs = bvecs
        self.model = DBSI_Autoencoder(len(bvals), bvals, bvecs).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
    def train_on_real_data(self, data_volume, mask, epochs=30, batch_size=2048):
        """
        Addestra il modello direttamente sui dati del paziente.
        """
        print(f"[Self-Supervised] Training on Real Data ({np.sum(mask)} voxels)...")
        
        # Estrai voxel validi e normalizza
        valid_data = data_volume[mask]
        b0_mean = np.mean(valid_data[:, self.bvals<50], axis=1, keepdims=True) + 1e-9
        valid_data = valid_data / b0_mean
        
        # Dataset PyTorch
        dataset = TensorDataset(torch.tensor(valid_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        loss_fn = nn.MSELoss()
        
        for epoch in tqdm(range(epochs), desc="Self-Supervised Training", file=sys.stdout):
            epoch_loss = 0
            for batch in loader:
                real_signal = batch[0].to(DEVICE)
                
                self.optimizer.zero_grad()
                
                # 1. Predici e Ricostruisci
                recon_signal, fractions, params = self.model(real_signal)
                
                # 2. Loss di Ricostruzione (Il segnale deve essere uguale)
                rec_loss = loss_fn(recon_signal, real_signal)
                
                # 3. Regolarizzazione Sparsità (Opzionale ma utile per DBSI)
                # Incoraggia frazioni a essere 0 o 1 (più nette), meno "mescolate"
                # L1 penalty sulle frazioni non aiuta molto perché sommano a 1.
                # Usiamo Entropia: -sum(p * log(p)) -> Minimizzare entropia = decisioni più nette
                entropy_loss = -torch.mean(torch.sum(fractions * torch.log(fractions + 1e-9), dim=1))
                
                # Totale: Ricostruzione + 0.01 * Entropia
                total_loss = rec_loss + 0.005 * entropy_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += rec_loss.item()
                
    def predict_volume(self, data_volume, mask):
        self.model.eval()
        X, Y, Z, N = data_volume.shape
        
        valid_data = data_volume[mask]
        b0_mean = np.mean(valid_data[:, self.bvals<50], axis=1, keepdims=True) + 1e-9
        valid_data = valid_data / b0_mean
        
        dataset = TensorDataset(torch.tensor(valid_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        frac_preds = []
        with torch.no_grad():
            for batch in loader:
                # L'output del decoder include le frazioni normalizzate
                _, fracs, _ = self.model.decoder(self.model.encoder(batch[0].to(DEVICE)))
                frac_preds.append(fracs.cpu().numpy())
                
        flat_fracs = np.concatenate(frac_preds, axis=0)
        
        # Ricostruisci mappe 3D
        maps = {}
        # Ordine nel PhysicsLayer: [res, hin, wat, fib]
        keys = ['ssl_restricted', 'ssl_hindered', 'ssl_water', 'ssl_fiber']
        for i, k in enumerate(keys):
            vol = np.zeros((X, Y, Z))
            vol[mask] = flat_fracs[:, i]
            maps[k] = vol
            
        return maps