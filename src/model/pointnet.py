from __future__ import annotations
from torch.nn import ReLU
import torch.nn as nn
import torch
import numpy as np

class PointNet(nn.Module):
    def __init__(self, in_channel:int, layers:List) -> None:
        
        super(PointNet, self).__init__()
        # currently only did for SSG because lazy
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        # appparently your in_channel needs to add 3?
        # anyway let's just assume we did not calculate ball query/FPS with only the xyz
        # but instead used all features (including xyz)
        # hence no need to add 3 to in_channel
        # in_channel += 3
        for i in layers:
            self.conv.append(nn.Conv2d(in_channel, i, 1))
            self.bn.append(nn.BatchNorm2d(i))
            in_channel = i
            
        self.relu = ReLU() # apparently you must instantiate ReLU here and call it later??!!

    def forward(self, input):
        out = input.permute(0, 3, 2, 1) # [B, d+C, nsample,npoint]

        for i in range(len(self.conv)):
            out = self.conv[i](out)
            out = self.bn[i](out)
            out = self.relu(out)

        # max pooling 
        out = torch.max(out, 2)[0]
        
        # permute to [B, npoint, d']
        out = out.permute(0, 2, 1)

        return out

if __name__ == "__main__":

    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 1000, 32, 3).astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X).to(device)

    model = PointNet(3, [32, 32, 64])
    model = model.to(device)
    
    summary(model, (1000,32,3))
    print ()
    print (model.forward(X).shape)