import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from losses import encoder_loss, generator_loss, discriminator_loss
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module, 
    optimizer: optim, 
    scheduler: lr_scheduler, 
    trainloader: torch.utils.data.DataLoader, 
) -> float:
    
    model.train()


    return 0