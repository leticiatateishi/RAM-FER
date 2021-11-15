import os
import argparse
import torch
from model import RecurrentAttention
from data_loader import get_data_loader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import argparse

import model

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--n_to_plot', type=int, default=25)
parser.add_argument('--n_rows', type=int, default=5)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_glimpses', type=int, default=6)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--n_patches', type=int, default=1)
parser.add_argument('--scale', type=float, default=2)
args = parser.parse_args()

assert args.n_to_plot % args.n_rows == 0 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

print(f'Device: {device}')



def sample_glimpses(model, iterator, num_glimpses, device='cpu'):

    model.eval()

    for images, _ in iterator:

        x_0 = images.to(device)
        x_0 = torch.reshape(x_0, (x_0.shape[0], 1, x_0.shape[1], x_0.shape[2]))
        batch_size = x_0.shape[0]

        l_t = torch.zeros([batch_size, 2], dtype=torch.float32).to(device)

        h_state = (torch.zeros(1, batch_size, 512).to(device), torch.zeros(1, batch_size, 512).to(device))

        locations = []
        for t in range(num_glimpses):
            locations.append(l_t)
            _, l_t, _, predicted, _, _, _ = model(x_0, l_t, h_state)
        
        loc = torch.stack((locations[0], locations[1], locations[2], locations[3], locations[4], locations[5], locations[6], locations[7]))

        predictions = torch.argmax(predicted, dim=-1)

        return images, predictions, loc


# Build the model
model = RecurrentAttention(
    args.patch_size,
    args.n_patches,
    args.scale
)
    
# Set the model to the device
model.to('cpu')

# Set the data loader
test_loader = get_data_loader(args.batch_size, is_train=False)[0]

# Set the model to be loaded
model_name = args.dir.split('/')[-1]
print(model_name)

# Set the folders
filename = model_name + "_checkpoint.tar"
output_path = os.path.join(args.dir)
checkpoint_path = os.path.join(output_path, 'checkpoint')
glimpse_path = os.path.join(output_path, 'glimpse')
heatmap_path = os.path.join(output_path, 'heatmap')
loss_path = os.path.join(output_path, 'loss')

checkpoint_path = os.path.join(checkpoint_path, filename)
print(checkpoint_path)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

images, predictions, locations = sample_glimpses(model, test_loader, args.n_glimpses, 'cpu')

#locations =  torch.FloatTensor(locations)
print(f'images {images.shape}, locations {locations.shape}')
batch_size, height, _ = images.shape
n_channels = 1

assert batch_size >= args.n_to_plot 

images = images[:args.n_to_plot]
predictions = predictions[:args.n_to_plot]
locations = locations[:,:args.n_to_plot]
print(locations.shape)

# convert locations from [-1, +1] to [0, height]
locations = (0.5 * ((locations + 1.0) * height))

fig, axes = plt.subplots(nrows=args.n_rows, ncols=args.n_to_plot//args.n_rows)
fig.tight_layout(pad=0.1)

images = images.cpu().numpy()
#images = np.transpose(images, [0, 2, 3, 1])
images = images.squeeze()

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap="Greys_r")
    xlabel = f'{predictions[i]}'
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])

def bounding_box(x, y, size, color='w'):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect

def update_image(i):
        color = 'r'
        location = locations[i,:]
        for j, ax in enumerate(axes.flat):
            while len(ax.patches) > 0:
                for p in ax.patches:
                    p.remove()
            loc = location[j]
            s = 1
            for k in range(args.n_patches):
                rect = bounding_box(
                    loc[0], loc[1], args.patch_size*s, color
                )
                ax.add_patch(rect)
                s = s * args.scale

anim = animation.FuncAnimation(
    fig, update_image, frames=args.n_glimpses, interval=1000, repeat=True)

print(anim.__class__)
name = f'{args.dir}/glimpses.mp4'
print(f'Saving to {name}')
anim.save(name)
    
