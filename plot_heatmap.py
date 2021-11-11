import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import pickle
import argparse
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.getcwd())

image_size_x = 48
image_size_y = 48

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.2)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax, orientation="horizontal")
    #cbar.ax.set_ylabel(cbarlabel, rotation=0, va="bottom")

    # We want to show all ticks...
    #ax.set_xticks(np.arange(data.shape[1]))
    #ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

denormalize = lambda x, coords : (0.5 * ((coords + 1.0) * x)).long()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--dir", type=str, required=True, help="path to the output execution")
    arg.add_argument("--glimpse", type=int, required=True, help="glimpse to plot")
    arg.add_argument("--train", type=str2bool, required=False, help="should open the train data")
    
    args = vars(arg.parse_args())
    
    return args["dir"], args["glimpse"], args["train"]


def plot(glimpses_array, plot_dir, name):
    
    # Separate each coordinate
    glimpses_array_x = glimpses_array[:, 0]
    glimpses_array_y = glimpses_array[:, 1]
    
    # Denormalize coordinates
    glimpses_array_x = denormalize(glimpses_array_x, image_size_x)
    glimpses_array_y = denormalize(glimpses_array_y, image_size_y)
    
    x = glimpses_array_x.cpu().data.numpy()
    y = glimpses_array_y.cpu().data.numpy()

    heatmap_data, xedges, yedges = np.histogram2d(x, y)
    
    fig, ax = plt.subplots(figsize=(15,7))
    
    row_names = list(range(1, 37))
    col_names = list(range(1, 123))
    
    im, _ = heatmap(heatmap_data.T, row_names, col_names, ax=ax, cmap='magma_r', cbarlabel="total glimpses", interpolation='spline16')
    
    #annotate_heatmap(im, valfmt="{x:.0f}", size=6, threshold=np.amax(heatmap_data)//3, textcolors=("red", "white"))

    plt.savefig(os.path.join("out", plot_dir, f'{name}'), orientation='landscape', dpi=100, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)


def main(plot_dir, glimpse, train):
    
    # Read the pickle files
    if train:
        
        # Get all the files inside the dir
        heatmaps_raw = [f.lower() for f in os.listdir(os.path.join(plot_dir)) if os.path.isfile(os.path.join(plot_dir, f))] 
        
        # Create the results folder
        if not os.path.exists(f'{plot_dir}/results'):
            os.makedirs(f'{plot_dir}/results')

        # For each image
        for i, filename in enumerate(reversed(sorted(heatmaps_raw, key=natural_keys))):
            
            if i % 20 == 0:
            
                glimpses_array = pickle.load(open(os.path.join("out", plot_dir, filename), "rb"))
                
                # Get the filename without extension
                name = filename.split('.')[0]
            
                if glimpse == 0:
                    glimpses_array = glimpses_array[1:].view(-1, 2)
                    name = f'all_{name}.png'
                else:
                    glimpses_array = glimpses_array[glimpse-1]
                    name = f'{glimpse}_{name}.png'
            
                # Plot the heatmap
                plot(glimpses_array, f"{plot_dir}/results", name)


    
    else:
        
        glimpses_array = pickle.load(open(os.path.join("out", plot_dir, f"glimpses_heatmap.p"), "rb"))
    
        if glimpse == 0:
            glimpses_array = glimpses_array.view(-1, 2)
            name = 'heatmap_all.pdf'
        else:
            glimpses_array = glimpses_array[glimpse-1]
            name = f'heatmap_{glimpse}.pdf'
    
        # Plot the heatmap
        plot(glimpses_array, f"{plot_dir}", name)
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(*args)