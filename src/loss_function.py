import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
import numpy as np
from torch import Tensor

def one_hot_encode(smnt: torch.Tensor, num_classes) -> torch.Tensor:
    
    B, C, H, W = smnt.shape    
    # change 255 to 0 since we don't care about the border
    smnt = torch.where(smnt == 255, 0, smnt)
        
    smnt_one_hot = one_hot(smnt.to(torch.int64), num_classes = num_classes + 1)
    
    # permute and reshape to get the correct shape
    smnt_one_hot = smnt_one_hot.permute(0, 1, 4, 2, 3)
    smnt_one_hot = smnt_one_hot.reshape(B, num_classes + 1, H, W)
    
    # remove the background class
    # smnt_one_hot = smnt_one_hot[:, :, 1:, :, :]

    return smnt_one_hot

def energy_loss(pred = torch.Tensor, ground_truth= torch.Tensor, weight = torch.Tensor, multiclass = False) -> torch.Tensor:
    """Energy loss
    Supposed to follow the following steps
    1. Softmax function along the channels
    2. Compute the cross entropy loss when compared to the ground truth
    3. Weighted sum of the loss

    But somehow torch.nn.CrossEntropyLoss() does all of this for you. So, we just need to call it

    Args:
        pred (_type_, optional): pred. 0 is always background. Hence, if you will have n_classes + 1 channels. Defaults to torch.Tensor.
        ground_truth (_type_, optional): Ground Truth. If the mask has 255 it is ignored. Needs to be int64 type. Defaults to torch.Tensor.

    Returns:
        torch.Tensor: _description_
    """
    if ground_truth.dtype != torch.int64:
        ground_truth = ground_truth.type(torch.int64)
        
    if multiclass:
        loss = torch.nn.CrossEntropyLoss(weight = weight)
    else:
        loss = torch.nn.BCEWithLogitsLoss(weight = weight)
        pred = pred.squeeze(1)
        ground_truth = ground_truth.type(torch.float32)


    return loss(pred, ground_truth.squeeze(1))

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    
    if multiclass:
        num_classes = input.size(1)
        input = prep_input(input)
        target = prep_target(target, num_classes = num_classes)

        # below is just playing with if you remove the background class in your dice loss
        input = input[:, 1:, :, :]
        target = target[:, 1:, :, :]
    
    return 1 - fn(input, target, reduce_batch_first=True)

def prep_input(input: Tensor):
    # Prepare input for dice loss
    input = torch.softmax(input, dim = 1).float()
    
    return input

def prep_target(target: Tensor, num_classes: int):
    # no need for the next two lines since we will preprocess it before that
    # target = target* 255
    # target = torch.where(target == 255, 0, target)

    target = target.to(torch.int64)
    target = one_hot(target, num_classes).squeeze(1).permute(0, 3, 1, 2).float()
    return target

if __name__ == "__main__":
    from src.model import UNet
    from src.dataset import get_load_data

    unet_model = UNet(num_classes=20)
    train, test = get_load_data(root = "../data", dataset = "VOCSegmentation", download = False)  
    img, smnt = train[0] 
    img = img.reshape(1, 3, 572, 572)

    smnt = smnt.resize((388, 388))

    # change smnt to torch.Tensor here
    smnt = torch.Tensor(np.asarray(smnt)).reshape(1, 1, 388, 388)

    pred = unet_model(img)

    loss = energy_loss(pred, smnt)