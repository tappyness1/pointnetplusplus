import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
from src.loss_function import energy_loss, dice_loss, dice_coeff, prep_input, prep_target
from sklearn.metrics import confusion_matrix

def get_accuracy(preds, ground_truth):
    ground_truth = ground_truth.squeeze(dim=1)
    preds = preds.argmax(dim=1)
    
    return (preds.flatten()==ground_truth.flatten()).float().mean()

def validation(model, val_set, cfg):
    """Simple validation workflow. Current implementation is for F1 score

    Args:
        model (_type_): _description_
        val_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()

    if cfg['train']['subset']:
        subset_indices = torch.randperm(len(val_set))[:cfg['train']['subset']]
        val_set = Subset(val_set, subset_indices)

    val_dataloader = DataLoader(val_set, batch_size=5, shuffle = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg['train']['loss_function'] == 'energy_loss' and cfg['train']['get_weights']:
    
        weights = torch.Tensor([1.33524479, 142.03737758, 354.33279529, 121.55201728,
                                170.52369266, 173.57602029,  59.18592147,  73.39980364,
                                39.04301533,  91.24823152, 124.53864632,  80.32893704,
                                62.08797479, 112.79122179,  92.20176115,  21.86262213,
                                161.68561906, 118.22250115,  72.47050034,  65.89660941,
                                116.10541954])
        weights = weights.to(device)

    else:
        weights = None

    model = model.to(device)
    dice_scores = []
    losses = []

    with tqdm(val_dataloader) as tepoch:

        for imgs, smnts in tepoch:
            
            with torch.no_grad():
                out = model(imgs.to(device))
            
            smnts *= 255
            smnts = torch.where(smnts==255, 0, smnts)
            smnts = smnts.to(device)


            if cfg['train']['multiclass']:
                dice_score = dice_coeff(prep_input(out), prep_target(smnts, cfg['train']['num_classes']))
            else:
                dice_score = dice_coeff(out, smnts)
        
            if cfg['train']['loss_function'] == 'energy_loss':
                loss = energy_loss(out, smnts, weight = weights, multiclass = cfg['train']['multiclass'])
            else:
                loss = dice_loss(out, smnts, multiclass = cfg['train']['multiclass']) 
            tepoch.set_postfix(dice_score=dice_score.item(), loss=loss.item())  
            losses.append(loss.item())
            dice_scores.append(dice_score.item())

    print (f"Dice Score: {sum(dice_scores)/len(dice_scores)}")
    print (f"Validation Loss: {sum(losses)/len(losses)}")


    return sum(dice_scores)/len(dice_scores), sum(losses)/len(losses)


if __name__ == "__main__":
    
    from src.dataset import get_load_data

    _, val_set = get_load_data(root = "../data", dataset = "VOCSegmentation")
    trained_model_path = "model_weights/model_weights.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(trained_model_path, map_location=torch.device(device))

    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': True, 
           'train': {"epochs": 20, 'lr': 1e-3, 
                     'weight_decay': 1e-8, 'momentum':0.999, 
                     'loss_function': 'dice_loss'}}
    validation(model, val_set, cfg)
            