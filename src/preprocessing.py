import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# make transform encoder
def ohe_transform(dataset: pd.DataFrame, subset: str, prefix: str, ohe: OneHotEncoder):
    
    """
    Transform a categorical column into one-hot encoded features using a fitted OneHotEncoder.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataset containing the categorical column to be encoded.

    subset : str
        The name of the categorical column in `dataset` that will be transformed.

    prefix : str
        The prefix that will be added to the generated encoded column names.
        Example: prefix="loan_grade" will produce columns such as
        "loan_grade_A", "loan_grade_B", etc.

    ohe : OneHotEncoder
        A fitted sklearn.preprocessing.OneHotEncoder object.
        The encoder must already be trained on the corresponding categorical feature.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where the selected categorical column has been replaced
        by its one-hot encoded representation.
    """

    # validation
    if not isinstance(dataset, pd.DataFrame):
        raise RuntimeError('Function ohe_transform: Dataset parameter must be in the dataframe type.')
    elif not isinstance(ohe, OneHotEncoder):
        raise RuntimeError('Function ohe_transform: OHE parameter must be in the OneHotEncoder type.')
    elif not isinstance(prefix, str):
        raise RuntimeError('Function ohe_transform: Prefix parameter must be in the str type.')
    elif not isinstance(subset, str):
        raise RuntimeError('Function ohe_transform: Subset parameter must be in the str type.')

    # check subset exists
    try:
        dataset.columns.tolist().index(subset)
    except:
        raise RuntimeError('Function ohe_transform: Subset parameter is string, but it could not be found in the column list on dataset.')
    
    print('Function ohe_transform: Parameters have been successfully validated.')
    
    # copy dataset
    dataset = dataset.copy()

    # columns before encoding
    print(f'Function ohe_transform: List of data columns before encoding: {dataset.columns.tolist()}')
    
    # create encoded column names
    col_names = [
        f'{prefix}_{col_name}' for col_name in ohe.categories_[0]
    ]

    # transform
    transformed = ohe.transform(dataset[[subset]]).toarray()
    encoded = pd.DataFrame(data=transformed,
                           columns=col_names,
                           index=dataset.index)
    
    # concat each encoded subset
    dataset = pd.concat([dataset, encoded], axis=1)

    # drop original column
    dataset.drop(columns=[subset], inplace=True)

    # columns after encoding
    print(f'Function ohe_transform: Column that have been successfully encoded: {dataset.columns.tolist()}')

    return dataset