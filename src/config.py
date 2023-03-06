import torch

class CFG:
    # thanks to awsaf for some amazing code style!!
    seed = 101
    exp_name = '2D_to_3D'
    comment = 'Trying out the 2D to 3D GAN from 3dgan.csail'
    model_name = 'VAE-3D-GAN'
    train_bs = 4
    valid_bs = 2*train_bs
    image_size = [256, 256]
    epochs = 5
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(30000/train_bs*epochs)+50
    T_0 = 25
    warmup_epochs = 0
    n_accumulate  = max(1, 32//train_bs)
    n_fold = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = None
    weight_decay = 1e-5