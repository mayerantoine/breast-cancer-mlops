import argparse
import sklearn
from azureml.core import Run
from azureml.core.model import Model as AMLModel
from azureml.core.resource_configuration import ResourceConfiguration

def main():
    """
    Register model to AML
    """

    parser = argparse.ArgumentParser("register")
    parser.add_argument("--model_file", type=str, help="model file")
    args = parser.parse_args()

    run = Run.get_context()
    ws = run.experiment.workspace
    # ds_tr = ws.get_default_datastore()

    model_path = args.model_file+"/cancer_model.pkl"
    print("model path:",model_path)

    AMLModel.register(workspace=ws,
                      model_name="breast-cancer",
                      model_path=model_path,
                      model_framework=AMLModel.Framework.SCIKITLEARN,
                      model_framework_version=sklearn.__version__,
                      resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                      description='Random forest classification model to predict breast cancer',
                       tags={'area': 'cancer', 'type': 'classification'})

if __name__ == '__main__':
    main()
