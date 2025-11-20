class DBSI_RegularizedMLP(nn.Module):
    """
    Macro-Compartment Architecture with Tighter Physical Constraints.
    Forces the Fiber to look like a Fiber (thin), preventing it from 
    cannibalizing the Hindered compartment.
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 20, dropout_rate: float = 0.1):
        super().__init__()
        self.n_iso = n_iso_bases
        
        # Masks for spectral aggregation
        grid = np.linspace(0, 3.0e-3, n_iso_bases)
        self.idx_res = torch.tensor(np.where(grid <= 0.3e-3)[0], dtype=torch.long)
        self.idx_hin = torch.tensor(np.where((grid > 0.3e-3) & (grid <= 2.0e-3))[0], dtype=torch.long)
        self.idx_wat = torch.tensor(np.where(grid > 2.0e-3)[0], dtype=torch.long)
        
        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_input_meas, 256),
            nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ELU()
        )
        
        # Heads
        self.head_fiber = nn.Linear(128, 1)       # Fiber Fraction
        self.head_iso_macro = nn.Linear(128, 3)   # [Res, Hin, Wat] prob
        self.head_iso_micro = nn.Linear(128, n_iso_bases) # Shape
        self.head_geom = nn.Linear(128, 4)        # [Theta, Phi, Dax, Drad]
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        # --- SMART INITIALIZATION ---
        self._init_weights()

    def _init_weights(self):
        # Initialize Linear layers standardly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # CUSTOM BIAS:
        # 1. Fiber: Start neutral (not negative anymore to avoid zeros)
        self.head_fiber.bias.data.fill_(0.0) 
        
        # 2. Macro Iso: Force network to look for HINDERED initially.
        # [Res, Hin, Wat] -> Bias [0, +1.0, 0] makes Hindered slightly preferred at start
        with torch.no_grad():
            self.head_iso_macro.bias[1] = 1.0 

    def forward(self, x):
        feat = self.backbone(x)
        
        # 1. Fiber Fraction
        f_fiber = self.sigmoid(self.head_fiber(feat)).squeeze(1)
        
        # 2. Isotropic Macro Probs
        iso_macro_probs = self.softmax(self.head_iso_macro(feat))
        
        f_iso_total = 1.0 - f_fiber
        
        # 3. Micro Shapes
        micro_logits = self.head_iso_micro(feat)
        
        # Restricted (Cell)
        res_dist = self.softmax(micro_logits[:, self.idx_res])
        w_res = res_dist * iso_macro_probs[:, 0:1] * f_iso_total.unsqueeze(1)
        
        # Hindered (Edema)
        hin_dist = self.softmax(micro_logits[:, self.idx_hin])
        w_hin = hin_dist * iso_macro_probs[:, 1:2] * f_iso_total.unsqueeze(1)
        
        # Water (CSF)
        wat_dist = self.softmax(micro_logits[:, self.idx_wat])
        w_wat = wat_dist * iso_macro_probs[:, 2:3] * f_iso_total.unsqueeze(1)
        
        # Concat
        f_iso_weights = torch.cat([w_res, w_hin, w_wat], dim=1)
        
        # 4. Geometry with TIGHTER CONSTRAINTS
        geom = self.head_geom(feat)
        theta = self.sigmoid(geom[:, 0]) * np.pi
        phi   = (self.sigmoid(geom[:, 1]) - 0.5) * 2 * np.pi
        
        # FIX: Tighter constraints to force separation
        # D_ax: Min 1.0 (Fibers are fast axially)
        d_ax  = self.sigmoid(geom[:, 2]) * 2.0e-3 + 1.0e-3  # Range [1.0, 3.0]
        
        # D_rad: Max 0.8 (Fibers are thin). 
        # If radial > 0.8, it MUST be modeled as Hindered Isotropic, not Fiber.
        d_rad = self.sigmoid(geom[:, 3]) * 0.8e-3           # Range [0.0, 0.8]
        
        # Logic constraint
        d_ax  = torch.max(d_ax, d_rad + 1e-6)

        return torch.cat([
            f_iso_weights, 
            f_fiber.unsqueeze(1), 
            theta.unsqueeze(1), 
            phi.unsqueeze(1), 
            d_ax.unsqueeze(1), 
            d_rad.unsqueeze(1)
        ], dim=1)