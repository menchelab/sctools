import logging

import statsmodels.formula.api as smf
import pandas as pd

from statsmodels.stats.multitest import fdrcorrection
from functools import partial


logging.basicConfig(format='%(asctime)s %(message)s')

def set_reference(data, categorical, reference):
    data = data.copy()
    series = data.loc[:, categorical].copy()
    data.drop(columns = [categorical], inplace = True)
    
    categories = list(series.unique())
    categories.remove(reference)
    data[categorical] = pd.Categorical(
        series,
        categories = [reference] + categories,
        ordered = True
    )

    return data


def summary(fit, drop_intercept = True):
    model_summary = []
    attrs = {
        'params': 'coefficients', 
        'bse': 'std_err', 
        'tvalues': 'tvals', 
        'pvalues': 'pvals'
    }
    for attr, attr_name in attrs.items():
        attr_values = getattr(fit, attr)
        attr_values.name = attr_name
        model_summary.append(attr_values)
    
    model_summary = pd.concat(model_summary, axis = 1)
    
    if drop_intercept:
        model_summary.drop(index = ['Intercept'], inplace = True)
        
    _, model_summary['padj'] = fdrcorrection(model_summary['pvals'])
        
    return model_summary
    

def compute_associations(data, score_column, categorical, reference_category = None, covariates = None):
    if reference_category:
        data = set_reference(
            data,
            categorical,
            reference_category
        )
    
    predictors = ' + '.join([categorical, *covariates]) if covariates else categorical
    model = smf.ols(
        f'{score_column} ~ {predictors}',
        data = data
    )
    regression = model.fit()
    return summary(regression)


# helper function for cleaner code
def compute_associations_per_group(data, *args, **kwargs):
    high_level_grouping = kwargs.pop('high_level_grouping')
    
    group_coeffs = {}
    for group_name, group_data in data.groupby(high_level_grouping, observed = True):
        group_coeffs[group_name] = compute_associations(group_data, *args, **kwargs)
        
    return pd.concat(group_coeffs, names = [high_level_grouping])
    

def category_effects_on_modules(
    adata, 
    gene_modules, 
    categorical,  
    scoring_func, 
    reference_category = None,
    high_level_grouping = None, 
    covariates = None,
    verbose = True
):
    """
    this function computes effect sizes of a given covariate on a set of gene modules.
    for this it takes the original object, a dictionary of gene lists, a categorical
    denoting a column in the anndata.obs dataframe containing the covariate in question
    as well as a scoring function to compute  a gene module score. Additionally, you can set
    a reference category for the categorical, a high level grouping
    in case you want to compute this for multiple groups (e.g. timepoints) and additional covariates
    for the model to consider (also need to be columns in the anndata.obs dataframe)
    
    :param adata:                  AnnData object containing the data to compute effect sizes from
    :param gene_modules:           dictionary of module_name: list(genes) pairs to compute effect sizes for
    :param categorical:            string denoting a column in AnnData.obs dataframe to compute effect sizes for
    :param scoring_func:           function computing module scores for the given modules. Must return a pandas.Series with index equal to AnnData.obs
    :param reference_category:     if set relevels the categorical to make sure the reference is the first level and thus used as a reference for computation
    :param high_level_grouping:    string denoting a column in the AnnData.obs dataframe used to split the data on effect sizes are computed separately for each group)
    :param covariates:             list of strings denoting columns in the AnnData.obs dataframe to use as additional covariates
    
    :return:                       pandas.DataFrame with the computed effect sizes + additional summary statistics
    """
    coeff_frames = {}
    association_func = (
        partial(
            compute_associations_per_group, 
            high_level_grouping = high_level_grouping
        ) 
        if high_level_grouping 
        else compute_associations
    )
    for module_name, module_genes in gene_modules.items():
        logging.info(f'computing gene scores on module {module_name}')
        scores = scoring_func(adata, module_genes)
        standardized_scores = (scores - scores.mean()) / scores.std()
        standardized_scores.name = 'standardized_score'
        obs_column_subset = [categorical]
        
        if covariates:
            obs_column_subset += covariates
            
        if high_level_grouping:
            obs_column_subset += [high_level_grouping]
            
        data = pd.concat(
            [
                standardized_scores, 
                adata.obs.loc[:, obs_column_subset]
            ],
            axis = 1
        )
        
        logging.info(f'estimating effects on module {module_name}')
        coefficients = association_func(
            data,
            'standardized_score',
            categorical,
            reference_category,
            covariates
        )
            
        coeff_frames[module_name] = coefficients.T
    
    return pd.concat(coeff_frames, names = ['modules'])
