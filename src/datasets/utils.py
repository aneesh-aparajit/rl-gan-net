import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import scipy.io as io


def load_data(pixel_path: Optional[str] = None, voxel_path: Optional[str] = None) -> Tuple[np.ndarray]:
    pixel, voxel = None, None
    if pixel_path:
        pixel = plt.imread(pixel_path)
    if voxel_path:
        voxel = io.loadmat(voxel_path)
    
    return pixel, voxel['voxel']
