import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from skimage.transform import resize

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        
        
class RecurrentAttention(nn.Module):
 
    def __init__(self, patch_size, num_patches, glimpse_scale):

        super().__init__()

        self.glimpse = GlimpseNetwork(patch_size, num_patches, glimpse_scale)
        self.core = CoreNetwork()
        self.locator = LocationNetwork()
        self.regressor = RegressorNetwork() 
        self.baseliner = BaselineNetwork()

    def forward(self, x_0, l_t, h_state):

        g_t, phi = self.glimpse(x_0, l_t)
    
        h_state = self.core(g_t, h_state)

        log_pi, l_t, entropy_pi = self.locator(h_state[0][-1])
        
        b_t = self.baseliner(h_state[0][-1]).squeeze()

        predicted = self.regressor(h_state[0][-1])

        return h_state, l_t, b_t, predicted, log_pi, phi, entropy_pi


class Retina:

    def __init__(self, patch_size, num_patches, glimpse_scale):
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.glimpse_scale = glimpse_scale

    def foveate(self, x, l):
  
        phi = []
        size = self.patch_size

        # extract num_patches patches of increasing size
        for i in range(self.num_patches):
            phi.append(self._extract_patch(x, l, size))
            size = int(self.glimpse_scale * size)

        # resize the patches to squares of size patch_size
        for i in range(1, len(phi)):
            batch = [resize(phi[i][j][0], (self.patch_size, self.patch_size), anti_aliasing=True) for j in range(phi[0].shape[0])]
            batch = np.array(batch)
            phi[i] = torch.unsqueeze(torch.Tensor(batch),1)

        # concatenate into a single tensor and flatten
        phi = torch.stack(phi)

        return phi

    def _extract_patch(self, x, l, size):

        B, C, H, W = x.shape

        start = self.denormalize(H, l)
        end = start + size

        # pad with zeros
        x = F.pad(x, (size//2, size//2, size//2, size//2))
        
        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):

            p = x[i, :, start[i, 1]:end[i, 1], start[i, 0]:end[i, 0]]

            patch.append(p)
        
        return torch.stack(patch)

    def denormalize(self, T, coords):
 
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):
  
        if (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T):
            return True
        return False


class GlimpseNetwork(nn.Module):
 
    def __init__(self, patch_size, num_patches, glimpse_scale):
        super().__init__()

        # Create the retina
        self.retina = Retina(patch_size, num_patches, glimpse_scale)
        
        # Image layers
        self.conv_1_1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn_1_1 = nn.BatchNorm2d(4)
        self.bn_1_2 = nn.BatchNorm2d(8)
        
        self.conv_2_1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv_2_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn_2_1 = nn.BatchNorm2d(4)
        self.bn_2_2 = nn.BatchNorm2d(8)
        
        self.conv_3_1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv_3_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn_3_1 = nn.BatchNorm2d(4)
        self.bn_3_2 = nn.BatchNorm2d(8)
        
        self.fc_x_1 = nn.Linear(800, 128)
        self.fc_x_2 = nn.Linear(800, 64)
        self.fc_x_3 = nn.Linear(800, 64)

        # Location layers
        self.fc_l_1 = nn.Linear(2, 256)
        self.fc_l_2 = nn.Linear(256, 256)

    def forward(self, x, l_t):
        
        # Generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t)
        x_1 = phi[0]
        x_2 = phi[1]
        x_3 = phi[2]
                
        # Use CNN in each glimpse scale
        x_1 = F.leaky_relu(self.bn_1_1(self.conv_1_1(x_1)), inplace=True)
        x_1 = F.leaky_relu(self.bn_1_2(self.conv_1_2(x_1)), inplace=True) 
        x_1 = x_1.view(phi[0].shape[0], -1)
        
        x_2 = F.leaky_relu(self.bn_2_1(self.conv_2_1(x_2)), inplace=True)
        x_2 = F.leaky_relu(self.bn_2_2(self.conv_2_2(x_2)), inplace=True) 
        x_2 = x_2.view(phi[1].shape[0], -1)
        
        x_3 = F.leaky_relu(self.bn_3_1(self.conv_3_1(x_3)), inplace=True)
        x_3 = F.leaky_relu(self.bn_3_2(self.conv_3_2(x_3)), inplace=True) 
        x_3 = x_3.view(phi[2].shape[0], -1)

        # Process the cnn output
        x_1 = self.fc_x_1(x_1)
        x_2 = self.fc_x_2(x_2)
        x_3 = self.fc_x_3(x_3)
        
        x_t = torch.cat([x_1, x_2, x_3], axis=1)
        
        # Flatten location vector
        l_t = l_t.view(l_t.size(0), -1)
        
        l_t = F.relu(self.fc_l_1(l_t), inplace=True)
        l_t = self.fc_l_2(l_t)

        # Concat the location and glimpses
        g_t = F.relu(x_t + l_t, inplace=True)

        return g_t, phi


class CoreNetwork(nn.Module):


    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(256, 512, 1)

    def forward(self, g_t, h_state):

        _, h_state = self.lstm(g_t.unsqueeze(0), h_state)
        
        return h_state


class RegressorNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_d_1 = nn.Linear(512, 256)
        self.fc_d_2 = nn.Linear(256, 7)

    def forward(self, h_t):
        
        d_t = F.relu(self.fc_d_1(h_t), inplace=True)
        d_t = torch.tanh(self.fc_d_2(d_t))

        # Clip between [-1, 1]
        d_t = torch.clamp(d_t, -1, 1)

        return d_t

class LocationNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.is_test = False

        self.fc_mu_1 = nn.Linear(512, 64)
        self.fc_mu_2 = nn.Linear(64, 32)
        self.fc_mu_3 = nn.Linear(32, 2)
        
        self.fc_std_1 = nn.Linear(512, 64)
        self.fc_std_2 = nn.Linear(64, 32)
        self.fc_std_3 = nn.Linear(32, 2)
         
        
    def forward(self, h_t):
               
        # Compute common layers
        mean = torch.tanh(self.fc_mu_1(h_t.detach()))
        mean = torch.tanh(self.fc_mu_2(mean))
        mean = torch.tanh(self.fc_mu_3(mean))
        
        # Compute the mean and logstd
        std = torch.tanh(self.fc_std_1(h_t.detach()))
        std = torch.tanh(self.fc_std_2(std))
        std = self.fc_std_3(std)
        
        std = torch.exp(std)
         
        # Create the normal dist
        normal = Normal(mean, std)
                
        x_t = normal.rsample()
        l_t = torch.tanh(x_t)
       
        log_pi = normal.log_prob(x_t)
        
        log_pi = log_pi.sum(dim=1)
        
        # Calc policy entropy
        entropy_pi = torch.sum(normal.entropy(), dim=1)
        
        if self.is_test:
            action = mean.detach()
        else:
            action = l_t.detach()
        
        return log_pi, action, entropy_pi.detach()


class BaselineNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(512, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 1)

    def forward(self, h_t):
        
        b_t = torch.relu(self.fc_1(h_t.detach()))
        b_t = torch.relu(self.fc_2(b_t))
        b_t = self.fc_3(b_t)
        
        return b_t