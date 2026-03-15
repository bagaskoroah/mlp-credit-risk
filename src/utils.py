import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import joblib
import yaml
import json

# define load config function
def load_config(config_path: str) -> dict:
    """
    Load the configuration file.

    Parameters:
    ----------
    config_path : str
        Configuration file location.
    
    Returns:
    -------
    config : dict
        Loaded configuration file.
    """
    try:
        # load config.yaml
        with open(config_path, 'r') as file:
           config = yaml.safe_load(file)
        # if .yaml is empty
        if config is None:
            raise ValueError('Configuration file is empty.')
    
    # yaml is not found in the directory
    except FileNotFoundError:
        raise FileNotFoundError('Config file not found in:', config_path)

    return config

# define update config function
def update_config(key, value, config, path_config):
    # add or update the config key and value
    config[key] = value

    # open config file
    with open(path_config, 'w') as file:
        yaml.dump(config, file)
    
    print(f'Config params updated! \nKey: {key} \nValue: {value}')

    # reload config
    config = load_config(config_path=path_config)

    return config

# define load json file function
def read_json_file(path: str) -> dict:
    """
    Load the json file.

    Parameter:
    ----------
    path: str
        Json file location.
    
    Return:
    ------
    json: dict
        Loaded json file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# define load data function
def load_data(fname: str) -> pd.DataFrame:
    '''
    Load dataframe and its data shape information.

    Param:
    fname <str> : raw dataset path to be loaded.

    Return:
    <pd.DataFrame> : loaded dataframe.
    '''
    data = pd.read_csv(fname)
    print('Data Shape:', data.shape)
    return data

# define split input output func
def split_input_output(data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:

    """
    Split a DataFrame into input features (X) and target variable (y).

    Params:
    data <pd.DataFrame> : Input dataset containing features and target column.
    target_col <str> : Name of the target column to be separated from the features.

    Returns:
    Tuple<pd.DataFrame, pd.Series>
        X : pd.DataFrame
            DataFrame containing input features.
        y : pd.Series
            Series containing the target variable.
    """

    X = data.drop(columns=target_col)
    y = data[target_col]

    print('Original data shape:', data.shape)
    print('X data shape:', X.shape)
    print('y data shape:', y.shape)

    return X, y

# define split train & test function
def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    '''
    Split features and target into training and testing sets.

    Params:
    X <pd.DataFrame> : Feature matrix.
    y <pd.Series> : Target variable.
    test_size <float> : Proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
    random_state <Optional[int], default=None> : Controls the shuffling applied to the data before splitting. Pass an integer for reproducible output.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train : pd.DataFrame
            Training feature set.
        X_test : pd.DataFrame
            Testing feature set.
        y_train : pd.Series
            Training target variable.
        y_test : pd.Series
            Testing target variable.
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print('X train shape:', X_train.shape)
    print('X test shape:', X_test.shape)
    print('y train shape:', y_train.shape)
    print('y test shape:', y_test.shape)
    return X_train, X_test, y_train, y_test

# define serialize function
def serialize_data(data: pd.DataFrame, path: str) -> None:
    """
    Serialize and save a Python object to disk using joblib.

    Params
    data <pd.DataFrame> : Python object to be serialized and saved.
    path <str> : File path where the serialized object will be stored.

    Return : None
    """
    joblib.dump(data, path)
    print('Saving object. . .')
    print(f'Your object has been successfully saved and stored into: {path}\n')

# define deserialize function
def deserialize_data(path: str) -> pd.DataFrame:
    """
    Load and deserialize a Python object from disk using joblib.

    Param
    path <str> : File path of the serialized object.

    Return
    data <pd.DataFrame> : The deserialized Python object loaded from disk.
    """
    print('Load object. . .')
    print(f'{path} has been successfully loaded!.')
    data = joblib.load(path)
    return data
    