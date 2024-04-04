import torch.nn as nn
from src.model import UNet
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf
from src.loss_function import energy_loss, dice_loss
from src.dataset import get_load_data
from src.dataloader import BasicDataset
import numpy as np
from src.validation import validation

def calc_weights(train_set):
    instances = []
    for i in range(len(train_set)):
        to_extend = train_set[i][1]
        to_extend *= 255
        instances.extend(to_extend.type(torch.int8).flatten().tolist())

    instances = np.array(instances)
    instances = np.where(instances == -1, 0, instances)

    counts = []
    for i in range(21):
        counts.append(np.count_nonzero(instances == i))
    counts = np.array(counts)
        
    weights = 1/ (counts / counts.sum())

    return torch.Tensor(weights)

def train(train_set, val_set, cfg, in_channels = 3, num_classes = 10):

    loss_function = None # using energy_loss instead
    
    # network = UNet(img_size = 572, num_classes = num_classes + 1)
    network = UNet(num_classes = num_classes)

    network.train()

    # optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    optimizer = optim.RMSprop(network.parameters(),
                            lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['weight_decay'], foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg['train']['loss_function'] == 'energy_loss' and cfg['train']['get_weights']:
       
       weights = calc_weights(train_set)
       weights = weights.to(device)

    else:
        weights = None

    network = network.to(device)

    if cfg['show_model_summary']:
        summary(network, (in_channels,512,512))

    if cfg['train']['subset']:
        subset_indices = torch.randperm(len(train_set))[:cfg['train']['subset']]
        train_set = Subset(train_set, subset_indices)
    
    train_dataloader = DataLoader(train_set, batch_size=6, shuffle = True)

    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, smnts in tepoch:
                # print (imgs.shape)

                # really only applicable for VOC segmentation data as the segmentations are class/255 for some unknown reason.

                smnts = smnts * 255
                smnts = torch.where(smnts == 255, 0, smnts)
                smnts = smnts.to(device)

                optimizer.zero_grad() 
                out = network(imgs.to(device))

                if cfg['train']['loss_function'] == 'energy_loss':
                    loss = energy_loss(out, smnts, weight = weights, multiclass = cfg['train']['multiclass'])
                else:
                    loss = dice_loss(out, smnts, multiclass = cfg['train']['multiclass'])
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
        _, loss = validation(network, val_set, cfg)
        scheduler.step(loss)
        network.train()
        
    print("training done")
    torch.save(network, cfg['save_model_path'])

    return network

if __name__ == "__main__":

    torch.manual_seed(42)

    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': True, 
           'train': {"epochs": 10, 'lr': 1e-3, 
                     'weight_decay': 1e-8, 'momentum':0.999, 
                     'loss_function': 'energy_loss', 
                     'subset': False, # set False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                     'multiclass': False, # set this to True if using VOC Segmentation
                     'get_weights': False, # set to true if using VOC Segmentation
                     'num_classes': 1} # if using VOC Segmentation, set to 21. If Carvana, use 1
            }
    
    train_set, val_set = get_load_data(root = "../data", dataset = "VOCSegmentation")
    train_set = BasicDataset(images_dir = "../data/carvana/train", mask_dir = "../data/carvana/train_masks")
    val_set = BasicDataset(images_dir = "../data/carvana/val", mask_dir = "../data/carvana/val_masks")
    train(train_set = train_set, val_set = val_set,  cfg = cfg, in_channels = 3, num_classes = cfg['train']['num_classes'])

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    
    # these are the hardcoded weight
    weights = torch.Tensor([1.33524479, 142.03737758, 354.33279529, 121.55201728,
       170.52369266, 173.57602029,  59.18592147,  73.39980364,
        39.04301533,  91.24823152, 124.53864632,  80.32893704,
        62.08797479, 112.79122179,  92.20176115,  21.86262213,
       161.68561906, 118.22250115,  72.47050034,  65.89660941,
       116.10541954])