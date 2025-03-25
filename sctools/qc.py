# import doubletdetection

import scanpy as sc
import pandas as pd
import numpy as np


# remove doubletdetection for now as dependency causes problems installing the package
# def detect_doublets(sample_id, sample_adata, n_jobs = 1, n_iters = 20, random_state = 0, **kwargs):
#     sc.pp.filter_genes(
#         sample_adata, 
#         min_cells = 1
#     )
#     
#     clf = doubletdetection.BoostClassifier(
#         n_iters = n_iters,
#         clustering_algorithm = "louvain",
#         standard_scaling = True,
#         pseudocount = 0.1,
#         n_jobs = n_jobs,
#         random_state = random_state,
#         verbose = False
#     )
#     doublet = clf.fit(sample_adata.X).predict(**kwargs)
#     doublet_frame = pd.DataFrame(
#         {
#             'doublet': doublet,
#             'doublet_score': clf.doublet_score()
#         },
#         index = sample_adata.obs.index
#     )
#     print(
#         '{}: predicted {} / {} cells as doublets'.format(
#             sample_id,
#             int(doublet.sum()),
#             len(doublet)
#         )
#     )
#     del clf, sample_adata
#    
#     return doublet_frame, int(doublet.sum()), len(doublet)


def get_nexpressed(adata):
    '''
    computes number of cells a gene is expressed in
    '''
    return np.array((adata.X > 0).sum(axis = 0)).flatten()


def ngenes_per_cell(adata)
    '''
    computes number of expressed genes per cell
    '''
    np.array((adata.X > 0).sum(axis = 1)).flatten()
    

def compute_qc_metrics(adata, mitochondrial_re = '^MT.', ribosomal_re = '^RP[SL]'):
    '''
    computes cell quality metrics like number of expressed features, total UMI count, percent mitochondrial UMIs
    and percent ribosomal UMIs.

    :param adata:               AnnData object to compute cell QC metrics for
    :param mitochondrial_re:    regular expression pattern to use for identification of mitochondrial genes
    :param ribosomal_re:        regular expression pattern to use for identification of ribosomal genes

    :return:                    None

    :Usage:
    ```
    from sctools import qc

    # by default this assumes human data so you may want to tune the regular expressions towards your model organism
    qc.compute_qc_metrics(adata)
    ```
    '''
    # flatten is needed due to csr_matrix.sum returning a numpy.matrix object
    # which cannot be broadcasted to obs frame
    adata.obs['nFeature_RNA'] = ngenes_per_cell(adata)
    adata.obs['nCount_RNA'] = np.array(adata.X.sum(axis = 1)).flatten()
    adata.obs['percent_mt'] = np.array(
        adata[:, adata.var.index.str.match(mitochondrial_re)].X.sum(axis = 1) / adata.X.sum(axis = 1) * 100
    ).flatten()

    adata.obs['percent_ribo'] = np.array(
        adata[:, adata.var.index.str.match(ribosomal_re)].X.sum(axis = 1) / adata.X.sum(axis = 1) * 100
    ).flatten()


def apply_qc_thresholds(adata, sample_id_column, sample_thresholds):
    '''
    takes a dictionary of thresholds per sample id and assesses whether a given cell passed all thresholds
    which is reported in the 'qc_pass' column added to adata.obs dataframe (requires compute_qc_metrics to be run first)

    :param adata:               AnnData object to assess QC passing on
    :param sample_id_column:    string denoting column containing sampleIDs to use for indexing 'sample_threshold' dictionary
    :param sample_threshold:    dictionary of dictionaries containing qc_metric: (lo, hi) thresholds per sampleID in 'sample_id_column'

    :return:                    None

    :Usage:
    ```
    from sctools import qc
    
    MIN_RNA_FEATURES = 750
    MAX_RNA_FEATURES = 6000
    MAX_PERCENT_MT = 15

    # define filtering thresholds
    qc_thresholds = {
        k: {
            'nFeature_RNA': (MIN_RNA_FEATURES, MAX_RNA_FEATURES), 
            'percent_mt': (0, MAX_PERCENT_MT), 
            'percent_ribo': (5, 100)
        } 
        for k 
        in adata.obs.sample_id
    }

    # compute QC passing cells
    # this requires compute_qc_metrics to be run first
    qc.apply_qc_thresholds(
        adata, 
        'sample_id', 
        qc_thresholds
    )

    # filter data
    adata = adata[adata.obs.qc_pass, :].copy()
    ```
    '''
    adata.obs['qc_pass'] = True
    for sample_id, thresholds in sample_thresholds.items():
        df = adata.obs.loc[adata.obs[sample_id_column] == sample_id, :]
        feature_qcs = []
        for feature, (lo, hi) in thresholds.items():
            feature_qcs.append(
                df[feature].apply(lambda x: x > lo and x < hi).values
            )
        
        qc_pass = np.all(
            np.vstack(feature_qcs),
            axis = 0
        )
        adata.obs.loc[adata.obs[sample_id_column] == sample_id, 'qc_pass'] = qc_pass
