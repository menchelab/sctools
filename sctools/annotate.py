import scanpy as sc
import pandas as pd
import numpy as np


def majority_vote(adata, prediction_col, clustering_resolution, majority_col_name = None):
    '''
    performs a majority annotation of cells based on their neighbourhood. In brief, the function
    performs Leiden clustering with `clustering_resolution` resolution and then annotates all cells
    in a given cluster with the annotation of the majority of the cells in the cluster. The reasoning
    behind this is that cells that are in close proximity are also similar. This implementation
    was adapted from the celltypist package.

    :param adata:                     AnnData object to perform annotation on. Has to have a computed kNN-graph (see sc.pp.neighbors)
    :param prediction_col:            string denoting the column to compute the majority annotation from              
    :param clustering_resolution:     resolution of the leiden clustering to use for majority annotation
    :param majority_col_name:         optional name of the new majority annotation column

    :return:                          None
    '''
    
    # partly taken from celltypist
    key_added = f'leiden_scvi_{clustering_resolution}'
    sc.tl.leiden(
        adata, 
        key_added = key_added,
        resolution = clustering_resolution
    )
    clustering = adata.obs.pop(key_added)
    votes = pd.crosstab(adata.obs[prediction_col], clustering)
    majority = votes.idxmax(axis=0)
    majority = majority[clustering].reset_index()
    majority.index = adata.obs.index
    
    majority_col_name = majority_col_name if majority_col_name else 'majority_voting'
    colnames = ['clustering', majority_col_name]
    majority.columns = colnames
    majority[majority_col_name] = majority[majority_col_name].astype('category')
    
    for col in colnames:
        if col in adata.obs.columns:
            adata.obs.pop(col)
    
    adata.obs = adata.obs.join(majority)


def annotate_adata_on_gene_hi_lo(adata, gene, resolution = 5):
    '''
    annotates cells based on the expression of a given gene in hi and lo expressing cells
    based on the median expression (i.e. gene expression of cell < median = lo else high)
    The raw annotation is then propagated to neighboring cells using a majority voting
    to mitigate possible dropouts

    :param adata:        AnnData object to perform the annotation for (needs to have a kNN-graph computed)
    :param gene:         name of the gene to annotate cells from
    :param resolution:   resolution of the leiden clustering used for majority propagation

    :return:             None
    '''
    expression_values = adata[:, gene].X.toarray().flatten()
    threshold = np.median(expression_values)
    status_column = f'{gene.lower()}_status'
    adata.obs[status_column] = np.select(
        [expression_values < threshold, expression_values >= threshold],
        [f'{gene}_lo', f'{gene}_hi']
    )
    majority_vote(
        adata,
        status_column,
        resolution,
        f'{status_column}_majority_vote'
    )
