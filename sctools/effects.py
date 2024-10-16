import statsmodels.formula.api as smf
import pandas as pd

from statsmodels.stats.multitest import fdrcorrection


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
    

def compute_associations(data, gene_score_column, categorical, reference_category = None, covariates = None):
    if reference_category:
        data = set_reference(
            data,
            categorical,
            reference_category
        )
    
    predictors = ' + '.join([categorical, *covariates]) if covariates else categorical
    model = smf.ols(
        f'{gene_score_column} ~ {predictors}',
        data = data
    )
    regression = model.fit()
    return summary(regression)
