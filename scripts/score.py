
import joblib
import json
import numpy as np
import os

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION)
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'cancer_model.pkl')
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)



input_sample = np.array([[0.393845668409139, -0.7457496952627328, -0.22189398515428446, -0.7676382304726264, -0.6999246324662508, -0.17745016145311038, 
-0.81484548808207, -0.770581761525873, -0.7189314051409194, 0.07367558407930598, -0.4665409990624931, -0.06425072325003647, -0.667435753669589, 
-0.10209930505592255, -0.2928749021935234, -0.18393920631817678, -0.8204798568998558, -0.6065568523529427, -0.395651155530143, 0.3134950272756633,
 -0.8678658791041564, -0.6897420117050609, -0.5598110994362666, -0.723009965205552, -0.6542518616646612, -0.5936861839456196,
 -0.9574791012332506, -0.9270503937797329, -0.7552651134179409, -0.2651797994882257, -1.0640128515011344]])
output_sample = np.array([0])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        print("input_data....")
        print(type(data))
        result = model.predict(data)
        # You can return any JSON-serializable object.
        return "here is your result = " + str(result)
    except Exception as e:
        error = str(e)
        return error
