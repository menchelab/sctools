import scvi

import scanpy as sc


def filter_too_few_cell_batches(adata, batch_key, min_cells = 5):
    '''
    computes the number of cells in each batch given in batch_key of adata.obs
    and removes all batches with fewer than `min_cells' cells. This function is mainly used
    for ensuring enough samples per batch for scVI.

    :param adata:      AnnData to filter
    :param batch_key:  column in adata.obs to use for counting the cells
    :param min_cells:  minimum number of cells per batch (default: 5)
    '''
    cells_per_batch = adata.obs.groupby(batch_key).count().iloc[:, 0]
    print(
        f'filtered the following batches < {min_cells} cells:\n', 
        cells_per_batch[cells_per_batch < min_cells]
    )
    enough_cells = adata.obs[batch_key].apply(
        lambda x: cells_per_batch[x] >= min_cells
    )
    return adata[enough_cells].copy()


def integrate_data_scvi(
    adata, 
    batch_key, 
    categorical_covariate_keys = None,
    continuous_covariate_keys = None,
    use_highly_variable_genes = True,
    n_top_genes = 4000,
    filter_small_batches = True,
    **kwargs
    
):
    '''
    integrate data using scVI based on batch_key and computes a umap on the computed latent representation. 
    This is basically set up as described in the scIB package https://github.com/theislab/scib/blob/main/scib/integration.py. 

    :param adata:                        AnnData object to perform the integration on
    :param batch_key:                    string denoting the column containing the batch information (e.g. sample/patient ID)
    :param categorical_covariate_keys:   list of strings denoting categorical covariates to consider for scVI (i.e. remove the effect of)
    :param continuous_covariate_keys:    list of strings denoting continuous covariates to consider for scVI (i.e. remove the effect of)
    :param use_highly_variable_genes:    whether to use all genes present or just the highly variable genes (default: True)
    :param n_top_genes:                  if to use highly variable genes how many to compute
    :param filter_small_batches:         whether to filter out small batches (i.e. batches with less then 5 cells)
    :param **kwargs:                     keyword arguments to pass to scVI.train method

    :return:                             dictionary with keys 'data' and 'model' containing the integrated data object as well as the trained scVI model

    :Usage:
    ```
    from sctools import integrate, plot
    
    # define a set of integration parameters for each dataset you want to integrate
    # this is useful for parameter sweeps or if different datasets require different parameterizations
    # in this case we simply have the same parameterization for all datasets
    integration_params = {
        k: {'kwargs': dict()} for k in adatas.keys()
    }
    
    integration_results = {}
    # adatas is a dictionary of datasets
    for key, adata in adatas.items():
        print(key)
        params = integration_params[key]
        integration_results[key] = integrate.integrate_data_scvi(
            adata.copy(),
            'sample_id',
            train_size = 1,
            **params['kwargs']
        )

        # directly writing to files in case something crashed so we at least don't lose the progress
        integration_results[key]['data'].write(
            f'../data/{key}.integrated.h5ad'
        )
    
        integration_results[key]['model'].save(
            f'../data/{key}.integration.scvi.model',
            overwrite = True
        )

    # inspect results
    # in this case we wanted to see if T-cells and especially FOXP3+ cells are clustering well
    # so for each integrated dataset we plot the same features on the UMAP
    # with the same kwargs for each
    fig, axs = plot.integrate.plot_integration_results(
        integration_results,
        ['sample_id', 'status', 'FOXP3', 'CD3D'],
        [
            dict(size = 20, vmax = None),
            dict(size = 20, vmax = None),
            dict(size = 20, vmax = 1),
            dict(size = 20, vmax = 5)
        ],
        data_key = 'data',
        legend_off = True
    )
    ```
    '''
    if filter_small_batches:
        adata = filter_too_few_cell_batches(adata, batch_key)

    else:
        adata = adata.copy()

    adata.layers['counts'] = adata.X.copy()
    adata.raw = adata
    
    if use_highly_variable_genes:
        print('computing highly variable genes')
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes = n_top_genes,
            layer = 'counts',
            subset = True,
            flavor = 'seurat_v3',
        )
        
    scvi.model.SCVI.setup_anndata(
        adata,
        layer = 'counts',
        batch_key = batch_key,
        categorical_covariate_keys = categorical_covariate_keys,
        continuous_covariate_keys = continuous_covariate_keys
    )
    # non default parameters from scVI tutorial and scIB github
    # see https://docs.scvi-tools.org/en/stable/tutorials/notebooks/harmonization.html
    # and https://github.com/theislab/scib/blob/main/scib/integration.py
    model = scvi.model.SCVI(
        adata,
        n_layers = 2,
        n_latent = 30,
        gene_likelihood = 'nb'
    )
    model.train(**kwargs)
    adata.obsm['X_scvi'] = model.get_latent_representation()
    
    print('compute umap from scvi embedding')
    sc.pp.neighbors(
        adata,
        use_rep = 'X_scvi'
    )
    sc.tl.umap(
        adata
    )
    
    return {'data': adata, 'model': model}
