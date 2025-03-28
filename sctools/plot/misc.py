import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc
import seaborn as sns


mpl.rcParams['pdf.fonttype'] = 42


def generate_and_save_for_figure(
    adata,
    columns_to_plot,
    scatter_save_file,
    legend_save_file,
    scatter_dpi = 1000,
    scatter_size = (5, 5),
    size = 10,
    edgecolor = 'none',
    linewidths = 1.5
):
    '''
    utility function to generate equal-sized plots of umaps by removing the legend saving it to an extra file

    :param adata:                  AnnData object to generate plots for
    :param columns_to_plot:        adata.obs columns to use for annotating the plots
    :param scatter_save_file:      path to the file to save the umap scatter to (recommend PNG)
    :param legend_save_file:       path to the file to save legend to (recommend PDF)
    :param scatter_dpi:            dot per inch to plot the umap PNG in
    :param scatter_size:           tuple denoting the size of the umap plot
    :param size:                   size of the scatter dots
    :param edgecolor:              color of the edge lines of the scatter dots
    :param linewidths:             with of the edge lines of the scatter dots

    :return:                       None

    :Usage:
    ```
    from sctools import plot

    # make patient_id colors persistent to have the same colors also for filtered datasets later on
    # e.g. T-cell subsets or similar
    patient_id_color_palette = {
        k: v for k, v 
        in zip(tmp.obs.patient_id.cat.categories, tmp.uns['patient_id_colors'])
    }

    # plot the thing
    plot.misc.generate_and_save_for_figure(
        adata,
        {
            'sample_id': (None, None),
            'patient_id': (patient_id_color_palette, None),
            'status': ('colorblind', None),
            'CD3D': (cmap, 3)
        },
        f'{plot_dir}/{k}' + '.raw.{0}.png',
        f'{plot_dir}/{k}' + '.raw.{0}.legend.pdf',
        size = 10,
        edgecolor = 'k',
        linewidths = 0.05
    )
    ```
    '''
    for col, (palette, vmax) in columns_to_plot.items():
        # generate figure and remove legend
        fig, ax = plt.subplots()
        width, height = scatter_size
        colorbar = False
        
        # if column is not in obs it is a gene
        if col not in adata.obs.columns:
            color_map = palette
            palette = None
            colorbar = True
        
        elif palette is None:
            palette = None
            color_map = None
        
        elif isinstance(palette, str):
            ncolors = adata.obs[col].nunique()
            palette = sns.color_palette(palette, ncolors)
            color_map = None
        
        elif isinstance(palette, dict):
            color_map = None
        
        sc.pl.umap(
            adata,
            color = col,
            frameon = False,
            palette = palette,
            color_map = color_map,
            colorbar_loc = None, # no colorbar to ensure w/h ratio
            size = size,
            vmax = vmax,
            ax = ax,
            show = False,
            edgecolor = edgecolor,
            linewidths = linewidths
        )
            
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        ax.title.set_visible(False)
        
        fig.set_figwidth(width)
        fig.set_figheight(height)
        fig.tight_layout()
        fig.savefig(scatter_save_file.format(col), dpi = scatter_dpi)
            
        
        if colorbar:
            plt.close(fig)
            fig, ax = plt.subplots()
            sc.pl.umap(
                adata,
                color = col,
                frameon = False,
                palette = palette,
                color_map = color_map,
                colorbar_loc = 'right',
                size = size,
                vmax = vmax,
                ax = ax,
                show = False
            )
        
            # remove scatters
            ax.clear()
            fig.savefig(legend_save_file.format(col))
        
        if not colorbar:
            # remove scatters
            ax.clear()
            ax.legend(handles, labels)
            fig.savefig(legend_save_file.format(col))
        
        plt.close(fig)
