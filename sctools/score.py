import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr


def bin_data(average_expression, nbins):
    '''
    assigns genes to one of nbins bins of equal numbers of genes based on their average expression. 

    :param average_expression:    np.array containing the average gene expression over all cells
    :param nbins:                 number of equal-sized bins to group genes into

    :return:                      np.array of length len(average_expression) containing the bin assignment for each gene

    :Usage:
    ```
    
    ```
    '''
    n_values = len(average_expression)
    num_vals_per_bin = n_values // nbins
    sorted_idx = np.argsort(average_expression)
    
    bin_assignments = np.zeros(shape = n_values, dtype = int)
    for bin_id, cut_idx in enumerate(range(0, n_values, num_vals_per_bin)):
        value_idx = sorted_idx[cut_idx: cut_idx + num_vals_per_bin]
        
        if len(value_idx) < num_vals_per_bin:
            bin_assignments[value_idx] = bin_id - 1
            break
            
        bin_assignments[value_idx] = bin_id
    
    return bin_assignments


def gene_module_score(adata, gene_list):
    '''
    computes the gene module score as implemented in Tirosh et al. Science 2016 (https://doi.org/10.1126/science.aad0501)
    In brief, this score is the difference between the average expression of genes in a module and a random sampled set of control genes
    (random sample adjusted for the magnitude of expression of each gene in gene module)

    :param adata:        AnnData object to compute score from
    :param gene_list:    list of genes in gene module to use for score computation (genes not in adata.var will be filtered out)

    :return:             np.array containing the computed score per cell

    :Usage:
    ```
    from sctools import score

    # make sure X contains log-normalized counts
    adata.X = adata.layers['counts']
    sc.pp.normalize_total(adata, target_sum = 1e4)
    sc.pp.log1p(adata)

    # read genes for set you want to score cells for
    genes = pd.read_csv(
        '../resource/fragility_score_genes.txt',
        sep = '\t'
    )

    # compute scores
    gene_scores = score.gene_module_score(
        adata, 
        genes.gene
    )
    ```
    '''
    genes_in_adata = set(adata.var.index.to_list())
    filtered_gene_list = list(
        set(gene_list) & genes_in_adata
    )
    
    average_expression = np.array(adata.X.mean(axis = 0)).flatten()
    gene_bins = pd.Series(
        bin_data(
            average_expression, 
            25
        ),
        index = adata.var.index
    )
    
    control_genes = set()
    for gene in filtered_gene_list:
        bin_id = gene_bins[gene]
        genes_in_bin = gene_bins[gene_bins == bin_id].index
        random_control_genes = np.random.choice(
            genes_in_bin, 
            size = 100, 
            replace = False
        )
        control_genes.update(random_control_genes)
    
    control_scores = np.array(adata[:, list(control_genes)].X.mean(axis = 1)).flatten()
    gene_list_scores = np.array(adata[:, filtered_gene_list].X.mean(axis = 1)).flatten()
    
    return gene_list_scores - control_scores 

# test code to ensure eigengenes has the same results as the R implementation
# import numpy as np
# import pandas as pd

# df = pd.read_csv(
#     '../data/wgcna_test.csv',
#     index_col = 0
# )

# np.random.seed = 12948
# random_gene_module = np.random.choice(df.columns[~df.columns.str.endswith('Rik')], size = 250, replace = False)

# with open('../data/random_gene_module.txt', 'w') as file:
#     for gene in random_gene_module:
#         file.write(gene + '\n')

# # R code
# library('WGCNA')
# df <- read.csv('../data/wgcna_test.csv', row.names = 1)
# colors <- rep('grey', dim(df)[2])
# names(colors) <- colnames(df)
# genes <- scan('../data/random_gene_module.txt', '\n')

# # this sometimes messes up due to some weird preceeding X of some genes in df columns upon import
# colors[genes] <- 'red'
# e <- moduleEigengenes(df, colors, excludeGrey = T, verbose = 10)
# # compare this to python output
# head(e$eigengenes)
# tail(e$eigengenes)


def ensure_one_dimensional(array):
    '''
    utility function to ensure the returned array is a 1D numpy.array and not a 
    2D numpy.matrix as is the case for aggregations of CSR matrix data
    '''
    if isinstance(array, np.matrix):
        array = np.array(array)
    
    else:
        array = array
        
    return array.flatten()


def align_to_expression(eigengene, bdata):
    '''
    utility function that ensures that orientation of eigengene follows average expression. 
    This is necessary due to sign indeterminacy of the SVD transformation
    '''
    average_expression = ensure_one_dimensional(bdata.X.sum(axis = 1))
    res = pearsonr(average_expression, eigengene)
    
    if res.statistic < 0:
        eigengene = -eigengene
    
    return eigengene


def module_eigengene(adata, genes):
    '''
    computes module eigenegenes for each cell in adata according to https://github.com/cran/WGCNA/blob/master/R/Functions.R
    using the truncated SVD implementation of sci-kit learn (seems to yield the same results as R as determined by running this 
    and then comparing to moduleEigengenes output in R based on the same data and selected genes; PCA would also work here 
    but shows slight differences in the values). In brief, this score is the sample weighting of the data according to the 
    singular vector with the corresponding to the highest singular value and basically gives a weighted average of gene expression
    in the module.

    :param adata:  AnnData to compute eigengenes for
    :param genes:  list of genes to use for eigengene computation (i.e. the genes contained in the module)

    :return:       pandas.Series containing eigengene score for each cell

    :Usage:
    ```
        from sctools import score

    # make sure X contains log-normalized counts
    # or normalization of choice
    adata.X = adata.layers['counts']
    sc.pp.normalize_total(adata, target_sum = 1e4)
    sc.pp.log1p(adata)

    # read genes for set you want to score cells for (or similar)
    genes = pd.read_csv(
        '../resource/fragility_score_genes.txt',
        sep = '\t'
    )

    # compute scores
    gene_scores = score.module_eigengene(
        adata, 
        genes.gene
    )
    
    ```
    '''
    bdata = adata[:, genes].copy()
    sc.pp.scale(bdata)
    svd = TruncatedSVD(n_components = 5)
    svd.fit(bdata.X.T)
    eigengene = align_to_expression(
        svd.components_[0, :],
        bdata
    )
    
    return pd.Series(eigengene, index = bdata.obs.index)
