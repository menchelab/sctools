import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42


def generate_qc_plots(
    axs, 
    df,
    qc_pass_idx, 
    thresholds = None
):
    '''
    utility function to generate qc plots to assess the impact of the QC filtering on the data

    :param axs:            2D numpy.array of Axes objects to generate the plots in
    :param df:             adata.obs of the dataset to assess QC for
    :param qc_pass_idx:    boolean numpy.array corresponding to df indicating if cell is filtered or not (see qc.compute_qc_metrics for details on this)
    :param thresholds:     optional dictionary of used threshold to compute qc_pass_idx (see qc.compute_qc_metrics for details on this)

    :return:               None
    '''
    datacols = ['nFeature_RNA', 'percent_mt', 'percent_ribo']
    hue = ['pass' if x else 'fail' for x in qc_pass_idx] if not all(qc_pass_idx) else None
    palette = {'pass': '#4B72B1', 'fail': 'red'} if hue else None
    for j, datacol in enumerate(datacols):
        sns.histplot(
            x = df.loc[:, datacol],
            ax = axs[0, j],
            hue = hue,
            palette = palette,
            kde = True,
            fill = True
        )
        if thresholds and datacol in thresholds:
            for position in thresholds[datacol]:
                if position:
                    axs[0, j].axvline(
                        position,
                        color = 'k',
                        linewidth = 1
                    )
    
    xy = [
        ('nCount_RNA', 'nFeature_RNA'),
        ('nFeature_RNA', 'percent_mt'),
        ('percent_mt', 'percent_ribo')
    ]
    for j, (xcol, ycol) in enumerate(xy): 
        sns.scatterplot(
            x = df.loc[:, xcol],
            y = df.loc[:, ycol],
            ax = axs[1, j],
            hue = hue,
            palette = palette,
            edgecolor = 'k',
            facecolor = None,
            color = None,
            alpha = 0.5
        )
        sns.kdeplot(
            x = df.loc[qc_pass_idx, xcol],
            y = df.loc[qc_pass_idx, ycol],
            ax = axs[1, j],
            color = 'lightblue'
        )
        
        if thresholds:
            for key, plotline in zip(
                [xcol, ycol],
                [axs[1, j].axvline, axs[1, j].axhline]
            ):
                if key in thresholds:
                    for position in thresholds[key]:
                        if position:
                            plotline(
                                position,
                                color = 'k',
                                linewidth = 1
                            )

                            
def plot_qc(
    adata,
    thresholds = None, 
    sample_id_column = None,
    sharex = False
):
    '''
    generates QC impact assessment plots for a given dataset containing multiple samples 
    (useful for QC filtering of datasets consisting of multiple datasets before integration)

    :param adata:           AnnData object containing the data to QC
    :param thresholds:      optional dictionary of used threshold to compute qc_pass_idx (see qc.compute_qc_metrics for details on this)
    :param sample_id_col:   string denoting the column containing the sample id info. if given qc is plotted for each sample separately else adata is considered in full
    :param sharex:          whether x axes of qc plot colums should be aligned (useful to directly compare samples)

    :return:                plt.Figure
    '''
    if not sample_id_column:
        fig, axs = plt.subplots(2, 3)
        generate_qc_plots(
            axs,
            adata.obs,
            qc_pass_idx = (
                adata.obs['qc_pass'] 
                if 'qc_pass' in adata.obs.columns 
                else [True] * adata.obs.shape[0]
            ),
            thresholds = thresholds
        )
        
    else:
        fig, axs = plt.subplots(
            adata.obs[sample_id_column].nunique(), 
            6, 
            sharex = 'col' if sharex else 'none'
        )
        for i, sample_id in enumerate(adata.obs[sample_id_column].unique()):
            tmp_df = adata[adata.obs[sample_id_column] == sample_id, :].obs
            generate_qc_plots(
                axs[i, :].reshape(2, 3),
                tmp_df,
                qc_pass_idx = (
                    tmp_df['qc_pass'] 
                    if 'qc_pass' in tmp_df.columns 
                    else [True] * tmp_df.shape[0]
                ),
                thresholds = thresholds[sample_id] if thresholds else None
            )
            axs[i, 0].set_ylabel(sample_id)
    
    return fig
