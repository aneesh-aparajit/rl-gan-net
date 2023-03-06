import torch
from torch.utils.data import Dataset
from typing import List
from src.config import CFG as cfg
import albumentations as A
from typing import Dict
from utils import load_data


class VAE3dGANDataset(Dataset):
    def __init__(self, in_paths: List[str], out_paths: List[str], stage: str = "train") -> None:
        super(VAE3dGANDataset, self).__init__()
        self.in_paths = in_paths
        self.out_paths = out_paths
        self.stage = stage
        
        if self.stage == "train":
            '''Ref: https://www.kaggle.com/code/alexeyolkhovikov/segformer-training'''
            self.transforms = A.Compose([
                A.augmentations.crops.RandomResizedCrop(height=cfg.image_size[0], width=cfg.image_size[1]),
                A.augmentations.Rotate(limit=90, p=0.2),
                A.augmentations.HorizontalFlip(p=0.2),
                A.augmentations.VerticalFlip(p=0.2),
                A.augmentations.transforms.ColorJitter(p=0.5),
                A.OneOf([
                    A.OpticalDistortion(p=0.2),
                    A.GridDistortion(p=0.2),
                    A.PiecewiseAffine(p=0.2)
                ], p=0.5),
                A.OneOf([
                    A.HueSaturationValue(10, 15, 10),
#                     A.CLAHE(clip_limit=4),
                    A.RandomBrightnessContrast()
                ], p=0.5),
                A.Normalize()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(cfg.image_size[0], cfg.image_size[1]),
                A.Normalize()
            ])
        
    
    def __len__(self) -> int:
        return len(self.in_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        in_path = self.in_paths[idx]
        out_path = self.out_paths[idx]
        
        pixels, voxels = load_data(in_path, out_path)
        
        # the transform is applied to only the 2d image
        pixels = self.transforms(image=pixels)
        
        return {
            "x": torch.tensor(pixels['image'], dtype=torch.float32).permute(2, 0, 1).to(cfg.device),
            "y": torch.tensor(voxels, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        }
