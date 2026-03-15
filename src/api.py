import utils
import preprocessing
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import sys
import uvicorn

# define project root
project_root = Path().resolve()
sys.path.insert(0, str(project_root))

# load config
config = utils.load_config(config_path=project_root/'config/config.yaml')

# load model config json
best_model_config = utils.read_json_file(path=project_root/'models/best_threshold.json')

# create API object
app = FastAPI()

# load model and ohe objects
best_model = utils.deserialize_data(path=project_root/config['best_model'])
ohe_default_on_file = utils.deserialize_data(path=project_root/config['ohe_default_on_file'])
ohe_home_ownership = utils.deserialize_data(path=project_root/config['ohe_home_ownership'])
ohe_loan_grade = utils.deserialize_data(path=project_root/config['ohe_loan_grade'])
ohe_loan_intent = utils.deserialize_data(path=project_root/config['ohe_loan_intent'])

# define input data structure
class DataAPI(BaseModel):
    ''' Represents the user input data structure. '''
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# define handlers
@app.post("/predict")
def predict(data: DataAPI):
    # convert DataAPI to Pandas DataFrame
    data = pd.DataFrame([data.dict()])

    # encoding str columns
    data = preprocessing.ohe_transform(
        dataset=data,
        subset=config['ohe_subset'][0],
        prefix=config['ohe_prefix'][0],
        ohe=ohe_home_ownership
    )

    data = preprocessing.ohe_transform(
        dataset=data,
        subset=config['ohe_subset'][1],
        prefix=config['ohe_prefix'][1],
        ohe=ohe_loan_intent
    )

    data = preprocessing.ohe_transform(
        dataset=data,
        subset=config['ohe_subset'][2],
        prefix=config['ohe_prefix'][2],
        ohe=ohe_loan_grade
    )

    data = preprocessing.ohe_transform(
        dataset=data,
        subset=config['ohe_subset'][3],
        prefix=config['ohe_prefix'][3],
        ohe=ohe_default_on_file
    )

    # predict data
    threshold = best_model_config['threshold']

    # predict probability
    proba = best_model.predict_proba(data)[:, 1][0]

    # apply threshold
    y_pred = int(proba >= threshold)

    if y_pred == 0:
        res = "Non Default"
    else:
        res = "Default"

    return {
        "prediction": y_pred,
        "probability": float(proba),
        "label": res
}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)