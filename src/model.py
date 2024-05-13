from __future__ import annotations

from src.pointnet import PointNet
from torch.nn import ReLU
import torch.nn as nn
import torch
import numpy as np
from pytorch3d.ops import sample_farthest_points, ball_query

class PNAbstractionSet(nn.Module):
    
    def __init__(self, K: int, radius: float, k_classes: int, in_channels: int, layers: List[int]) -> None:
        """PointNet Abstraction Set 

        Args:
            K (int): K groups for the ball query
            radius (float): radius for the ball query
            k_classes (int): number of classes. This is for selecting the number of points to sample in ball query
            in_channels (int): input's in channels
            layers (List[int]): layers for the MLP/1D convolution
        """

        super(PNAbstractionSet, self).__init__()

        self.pointnet = PointNet(in_channels, layers)
        self.K = K
        self.radius = radius
        self.k_classes = k_classes

    def forward(self, input: torch.Tensor) -> torch.Tensor :
        # input shape: B x npoints x (d+C)

        # FPS to get the centroids, B x K x (d+C)
        # in the paper it is stated to use the positions
        # in practice, all the features are used
        selected_points, _ = sample_farthest_points(input, K = self.K)
        selected_points_xyz = selected_points[:,:,:3] # only the xyz coordinates
        
        # ball query to get the grouped points
        # output is B x N' x nsamples(K but means number of samples per centroid) x (d+C) 
        # where N' refers to the number of centroids 
        _, _, grouped_points = ball_query(selected_points, input, K = self.k_classes, radius = self.radius)

        out = self.pointnet(grouped_points)
        out = torch.cat((selected_points_xyz, out), dim = 2) # in original code, they cat xyz at the back

        return out
    
def interpolate(points1:torch.Tensor, points2:torch.Tensor, k = 3) -> torch.Tensor:
    """Interpolate the features to the higher layer using inverse distance weighting

    Args:
        points1 (torch.Tensor): The l-1 layer output with d + C1 features. Not as many features, but more points
        points2 (torch.Tensor): The l layer output with d+C2 features. More features, but less points
        k (int, optional): k neighbours for interpolating the features. Defaults to 3.

    Returns:
        torch.Tensor: The interpolated features with the same number of points as points1, but with combined features of points1 and points2 (d + C1 + C2) 
    """
    xyz1, xyz2 = points1[:,:,:3], points2[:,:,:3]
    features1, features2 = points1[:,:,3:], points2[:,:,3:]
    distances = torch.cdist(xyz1, xyz2, p = 2) # B x N1 x N2
    topk_results = torch.topk(distances, k = k, dim = 2, largest = False) # call .values or .indices
    topk_weights = 1 / (topk_results.values + 1e-8) # add 1e-8 to prevent division by zero
    # normaliser = topk_weights.sum(dim = 2, keepdim = True)
    top_k_norm_weights = topk_weights / topk_weights.sum(dim = 2, keepdim = True)

    # multiply the weights with the features. Need to use the topk indices to fish out the correct points
    # Get the indices for batch and row dimension. Thanks CoPilot for the suggestion
    batch_indices = torch.arange(topk_results.indices.shape[0]).view(-1, 1).expand(-1, topk_results.indices.shape[1]*k).flatten()
    # row_indices = torch.arange(topk_results.indices.shape[1]).repeat(topk_results.indices.shape[0])

    # Use the indices to get the features
    feat = features2[batch_indices, topk_results.indices.flatten(), :]
    
    # multiply the features with the normalised weights
    feat = torch.mul(feat, top_k_norm_weights.flatten().unsqueeze(-1))
    feat = feat.reshape(-1, k, features2.shape[2])
    # sum along dim = 1 (since we reshaped to (BxN1, k, C)
    feat = feat.sum(dim=1)
    # Reshape the interpolated features to match the original shape
    interpolated_features = feat.view(features1.shape[0], features1.shape[1], features2.shape[2])

    # finally, we cat the interpolated features with the "original" features
    interpolated_points = torch.cat((points1, interpolated_features), dim = 2)

    # old method - double for loop
    # interpolated_features = torch.zeros(1, 512)
    # for batch in range(topk_results.indices.shape[0]):
    #     for row in range(topk_results.indices.shape[1]):
    #         # print (features2[batch, topk_results.indices[batch,row,:], :].shape)
    #         feat = features2[batch, topk_results.indices[batch,row,:], :]
    #         feat = torch.mul(feat, top_k_norm_weights[batch, row, :].unsqueeze(-1))
    #         feat = feat.sum(dim = 0).reshape(1,512)
    #         interpolated_features = torch.cat((interpolated_features, feat))
    # interpolated_features = interpolated_features[1:,:]
    # interpolated_features = interpolated_features.view(features1.shape[0],features1.shape[1], features2.shape[2])
  
    return interpolated_points

class PointFeaturePropagation(nn.Module):
    def __init__(self, in_channel, layers) -> None:
        
        super(PointFeaturePropagation, self).__init__()
        self.pointnet = PointNet(in_channel, layers)
        
    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:

        interpolated_points = interpolate(points1, points2)        
        interpolated_xyz = interpolated_points[:,:,:3]
        interpolated_features = interpolated_points[:,:,3:]
        interpolated_features = interpolated_features.unsqueeze(2) # [B, d+C, 1, npoint]
        out = self.pointnet(interpolated_features)
        out = torch.cat((interpolated_xyz, out), dim = 2)

        return out

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes:int, network_type:str= "Segmentation") -> None:
        super(PointNetPlusPlus, self).__init__()
        
        self.K = num_classes # in the paper it says K is num classes, but they could be talking nonsense again

        # radius to be set to increase 2x at each successive set
        self.pn_abstraction_set_1 = PNAbstractionSet(K = 1024, radius = 0.1, k_classes = self.K, in_channels = 3, layers = [32, 32, 64])
        self.pn_abstraction_set_2 = PNAbstractionSet(K = 256, radius = 0.2, k_classes = self.K, in_channels = 64+3, layers = [64, 64, 128])
        self.pn_abstraction_set_3 = PNAbstractionSet(K = 64, radius = 0.4, k_classes = self.K, in_channels = 128+3, layers = [128, 128, 256])
        self.pn_abstraction_set_4 = PNAbstractionSet(K = 16, radius = 0.8, k_classes = self.K, in_channels = 256+3, layers = [256, 256, 512])
        
        # the feature propagation section
        self.pn_feature_propagation_1 = PointFeaturePropagation(in_channel = 768, layers = [256, 256])
        self.pn_feature_propagation_2 = PointFeaturePropagation(in_channel = 384, layers = [256, 256])
        self.pn_feature_propagation_3 = PointFeaturePropagation(in_channel = 320, layers = [256, 128])
        self.pn_feature_propagation_4 = PointFeaturePropagation(in_channel = 128, layers = [128, 128, 128])

        self.dropout = nn.Dropout(p = 0.5)
        self.linear_1 = nn.Conv1d(128, 128, kernel_size = 1)
        self.linear_2 = nn.Conv1d(128, num_classes, kernel_size = 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = ReLU()

    def forward(self, input):
        skip_link_0 = input.clone()
        out = self.pn_abstraction_set_1(input) # end up with B x (3+64) x 1024, 
        skip_link_1 = out.clone()
        out = self.pn_abstraction_set_2(out) # end up with B x (3 + 128) x 256
        skip_link_2 = out.clone()
        out = self.pn_abstraction_set_3(out) # end up with B x (3 + 256) x 64
        skip_link_3 = out.clone() 
        out = self.pn_abstraction_set_4(out) # end up with B x 515 x 16, where N = 16, and d+ C = 3+515

        # with interpolated features
        out = self.pn_feature_propagation_1(skip_link_3, out)
        out = self.pn_feature_propagation_2(skip_link_2, out)
        out = self.pn_feature_propagation_3(skip_link_1, out)
        out = self.pn_feature_propagation_4(skip_link_0, out)

        out_features = out[:,:,3:]
        out_pos = out[:,:,:3]
        out_features = out_features.permute(0, 2, 1)
        out = self.relu(self.bn1(self.linear_1(out_features)))
        out = self.linear_2(out_features) # get them logits while hot
        out_features = out.permute(0, 2, 1)
        # out = torch.cat((out_pos, out), dim = 2)
        
        return out_pos, out_features

if __name__ == "__main__":

    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 2000, 3).astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X).to(device)

    model = PointNetPlusPlus(num_classes=10, network_type = "Segmentation")
    model = model.to(device)
    
    # summary(model, (1000,3)) # not while there's no parameters!
    print ()
    # here we will do a model.forward and then get the classes using log_softmax. The loss function given was NLLLoss
    out_pos, logits = model(X)
    print (logits)
    log_softmax = nn.LogSoftmax(dim = 2)
    out_classes = log_softmax(logits).argmax(dim = 2)
    print (out_pos.shape, logits.shape, out_classes.shape)