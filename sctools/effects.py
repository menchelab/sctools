import statsmodels.formula.api as smf
import pandas as pd

from statsmodels.stats.multitest import fdrcorrection


def set_reference_genotype(data, genotype_column, reference_genotype):
    data = data.copy()
    genotype_series = data.loc[:, genotype_column].copy()
    data.drop(columns = [genotype_column], inplace = True)
    
    genotypes = list(genotype_series.unique())
    genotypes.remove(reference_genotype)
    data[genotype_column] = pd.Categorical(
        genotype_series,
        categories = [reference_genotype] + genotypes,
        ordered = True
    )

    return data


def summary(fit):
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
    
    return pd.concat(model_summary, axis = 1)
    

def compute_associations(data, gene_score_column, genotype_column, reference_genotype, covariates = None):
    data = set_reference_genotype(
        data,
        genotype_column,
        reference_genotype
    )
    predictors = ' + '.join([genotype_column, *covariates]) if covariates else genotype_column
    model = smf.ols(
        f'{gene_score_column} ~ {predictors}',
        data = data
    )
    regression = model.fit()
    coefficients = summary(regression)
    _, coefficients['padj'] = fdrcorrection(coefficients['pvals'])
    return coefficients
