# dbsi_toolbox/common.py
import numpy as np
from dataclasses import dataclass

@dataclass
class DBSIParams:
    """
    Container for DBSI results for a single voxel, shared across all solvers.
    """
    # Isotropic Fractions
    f_restricted: float      # Restricted fraction (cells, inflammation)
    f_hindered: float        # Hindered fraction (extracellular space)
    f_water: float           # Free water fraction (CSF, edema)
    
    # Anisotropic Fraction
    f_fiber: float           # Total fiber fraction
    
    # Fiber Properties
    fiber_dir: np.ndarray    # Dominant fiber direction (x, y, z)
    axial_diffusivity: float
    radial_diffusivity: float
    
    # Fitting Quality
    r_squared: float

    @property
    def f_iso_total(self) -> float:
        """Returns the sum of all isotropic fractions."""
        return self.f_restricted + self.f_hindered + self.f_water

    @property
    def fiber_density(self) -> float:
        """Returns the normalized fiber density (Fiber Fraction)."""
        total = self.f_fiber + self.f_iso_total
        return self.f_fiber / total if total > 0 else 0.0