import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib import gridspec


def normalize_size(size_data, min_size, max_size, size_scale = 25):
    '''
    utility function to normalize sizes into a given scale between 0 and size_scale

    :param size_data:    2D np.array or similar containing values indicating the size of dots
    :param min_size:     minimum size to consider in the norm
    :param max_size:     maximum size to consider in the norm
    :param size_scale:   rescaling factor for the resulting norm

    :return:             normalized and rescaled size values
    '''
    norm = Normalize(min_size, max_size, clip = True)
    normalized_sizes = np.zeros(shape = size_data.shape)
    for i, row in enumerate(size_data.values):
        for j, size in enumerate(row):
            normalized_sizes[i, j] = norm(size)
    
    return normalized_sizes * size_scale


def hide_spines(ax):
    '''
    hides all spines of the given Axes object
    '''
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(False)


def add_dotsize_legend(min_size, max_size, size_scale, ax, n):
    '''
    adds a legend of dot sizes to the given Axes object

    :param min_size:        minimum dot size
    :param max_size:        maximum dot size
    :param size_scale:      dot size rescaler
    :param ax:              Axes object to plot the legend to
    :param n:               number of sizes to show in the legend

    :return:                None
    '''
    legend_sizes = np.linspace(min_size, max_size, n)
    
    if legend_sizes[0] == 0:
        legend_sizes = legend_sizes[1:]
        n = n - 1
        
    ymin, ymax = ax.get_ylim()
    xmin, xmax = (0, 100) #ax.get_xlim()
    x = [(xmax - xmin) / 2 for _ in legend_sizes]
    y = np.linspace(ymin, ymax, n)
    ax.scatter(
        x, y, 
        s = legend_sizes,
        c = 'grey'
    )
    hide_spines(ax)
    ax.set_xticks([])
    ax.set_yticks(y)
    ax.set_yticklabels(legend_sizes.astype(int))
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")


def dotplot(
    color_data, 
    size_data, 
    colormap = 'coolwarm', 
    vmin = None, 
    vmax = None, 
    size_norm = None,
    indicator_threshold = None, 
    indicate_from = 'size',
    indicator_color = 'k',
    row_order = None,
    col_order = None,
    size_scale = 25,
    figsize = (5, 10),
    cbar_label = None,
    dotsize_label = None
):
    '''
    creates a dotplot similar to Fig2B of https://doi.org/10.1126/science.aaz6063 

    :param color_data:                2D numpy.array or pandas.DataFrame containing data to use for coloring the dots
    :param size_data:                 2D numpy.array or pandas.DataFrame containing data to use for dot size
    :param colormap:                  colormap to use for coloring the dots (either a string for named colormap or a colormap object)
    :param vmin:                      minimum value of the colormap (values lower will be clipped) if not given uses minimum of color_data
    :param vmax:                      maximum value of the colormap (values higher will be clipped) if not given uses maximum of color_data
    :param size_norm:                 tuple of min and max value of the dotsizes, if given, values of size_data will be mapped to this scale, values outsize will be clipped
    :param indicator_threshold:       threshold value to use for indicating dots with a outline of color 'indicator_color'
    :param indicate_from:             indicates which data to compare to the theshold for indication (either 'size' or 'color')
    :param indicator_color:           color to use for indication
    :param row_order:                 row index specifying the order to plot rows in (useful to align plot with other plots e.g. seaborn.clustermap)
    :param col_order:                 column index specifying the order to plot columns in (useful to align plot with other plots e.g. seaborn.clustermap)
    :param size_scale:                rescale factor for dot sizes (mainly for aesthetic purposes)
    :param figsize:                   tuple of (figwidth, figheight) specifying the dimensions of the generated figure
    :param cbar_label:                label to use for the colorbar
    :param dotsize_label:             label to use for the dot size legend

    :return:                          plt.Figure
    '''
    num_rows, num_cols = color_data.shape
    xs = np.linspace(0, num_cols, num_cols)
    ys = np.linspace(0, num_rows, num_rows)
    xv, yv = np.meshgrid(xs, ys)
    
    if isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
    
    else:
        cmap = colormap
        
    if isinstance(row_order, type(None)):
        row_order = list(range(color_data.shape[0]))
        
    if isinstance(col_order, type(None)):
        col_order = list(range(color_data.shape[1]))
        
    if not size_norm:
        size_norm = (0, np.floor(size_data.max()))
        
    normed_sizes = normalize_size(size_data, *size_norm, size_scale)
    min_color = vmin if vmin else color_data.min()
    max_color = vmax if vmax else color_data.max()
        
    fig = plt.figure()
    gridrows = 10
    gs = gridspec.GridSpec(
        gridrows, 2,
        width_ratios = [0.95, 0.05],
        height_ratios = np.repeat(1/gridrows, gridrows)
    )
    dotplot = fig.add_subplot(gs[:, 0])
    dotplot.scatter(
        xv,
        yv,
        s = normed_sizes[row_order, :],
        c = color_data.values[row_order, :],
        vmin = min_color, 
        vmax = max_color,
        cmap = cmap
    )
    if indicator_threshold:
        indicator_data = color_data if indicate_from == 'color' else size_data
        indicate_idx = np.where(indicator_data.values[row_order, :][:, col_order] > indicator_threshold)
        dotplot.scatter(
            xv[indicate_idx],
            yv[indicate_idx],
            s = normed_sizes[row_order, :][indicate_idx],
            facecolors = 'none',
            edgecolors = indicator_color,
            linewidth = 0.5
        )
    
    dotplot.set_xlim(xs.min() - 1, xs.max() + 1)
    dotplot.set_ylim(ys.min() - 1, ys.max() + 1)
    
    hide_spines(dotplot)
        
    row_labels = color_data.index[row_order]
    col_labels = color_data.columns[col_order]
    
    dotplot.set_yticks(ys)
    dotplot.set_yticklabels(row_labels)
    
    dotplot.set_xticks(xs)
    dotplot.set_xticklabels(col_labels, rotation = 90)
    
    cbar = fig.add_subplot(gs[:2, 1])
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm = Normalize(min_color, max_color, clip = True),
            cmap = cmap
        ),
        cax = cbar,
        label = cbar_label
    )
    
    dotsizes = fig.add_subplot(gs[-1, 1])
    min_size, max_size = size_norm
    add_dotsize_legend(
        min_size,
        max_size,
        size_scale,
        dotsizes,
        n = int(max_size / 10 + 1)
    )
    dotsizes.set_ylabel(dotsize_label)
    fig.set_figwidth(figsize[0] + 2)
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    return fig
