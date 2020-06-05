import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        
        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')
            
        self.main = nn.Conv2d(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        output = self.main(x)
        output = self.activation(output)
        return output
        
        
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        # encoder functions
        self.enc = nn.ModuleList()
        nb_channels_enc = [2, 32, 32, 32, 32]
        for i in range(len(nb_channels_enc)-1):
            self.enc.append(ConvBlock(nb_channels_enc[i], nb_channels_enc[i+1], 2)) # 2 => 32 => 32 => 32 => 32
        
        # decoder functions
        self.dec = nn.ModuleList()
        nb_channels_dec = [32, 32, 32, 32, 32, 16]
        self.dec.append(ConvBlock(nb_channels_enc[-1], nb_channels_dec[0]))                     # 32 => 32
        self.dec.append(ConvBlock(nb_channels_dec[0] * 2, nb_channels_dec[1]))                  # 64 => 32
        self.dec.append(ConvBlock(nb_channels_dec[1] * 2, nb_channels_dec[2]))                  # 64 => 32
        self.dec.append(ConvBlock(nb_channels_dec[2] + nb_channels_enc[1], nb_channels_dec[3])) # 64 => 32
        self.dec.append(ConvBlock(nb_channels_dec[3], nb_channels_dec[4]))                      # 32 => 32
        self.dec.append(ConvBlock(nb_channels_dec[4] + 2, nb_channels_dec[5]))                  # 34 => 16
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    
    def forward(self, x):
        
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))
            
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i+2)]], dim=1)
            
        y = self.dec[3](y)
        y = self.dec[4](y)
        
        y = self.upsample(y)
        y = torch.cat([y, x_enc[0]], dim=1)
        y = self.dec[5](y)
        
        return y
        
    
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        
        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode
        
    def forward(self, src, flow):
        
        new_locs = self.grid + flow
        
        shape = flow.shape[2:]
        
        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
            
        new_locs = new_locs.permute(0, 2, 3, 1) 
        new_locs = new_locs[..., [1,0]]
        
        return nnf.grid_sample(src, new_locs, mode=self.mode)
    
####### Voxelmorph network (1st network)#######
class NetMain(nn.Module):
    def __init__(self, size):
        super(NetMain, self).__init__()
        
        self.unet_model = Unet()
        
        self.flow = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        
        self.spatial_transform = SpatialTransformer(size)
        
    def forward(self, src, tgt):
        
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        y = self.spatial_transform(src, flow)
        
        return y, flow
        
###############################################   
     
        
        
######### Affine network (2nd network)#########
class AffineNet(nn.Module):
    def __init__(self):
        super(AffineNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(48, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 80, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(80, 100, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(100, 120, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(240, 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),     
            nn.Linear(50, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 240)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = nnf.affine_grid(theta, x[:, None, 0, ...].size())
        x = nnf.grid_sample(x[:, None, 0, ...], grid)

        return x, grid
    
    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)

        # transform the input
        x = self.stn(x)

        return x
        
###############################################