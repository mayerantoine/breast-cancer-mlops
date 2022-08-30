
import azureml
import os
import sklearn
from azureml.core import  Workspace
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration,DockerConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Workspace, Experiment, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig, Environment
from azureml.widgets import RunDetails
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData


def get_aml_compute(workspace):
    clustername = 'StandardDS12CPU'
    is_new_cluster = False
    try:
        aml_compute = ComputeTarget(workspace = workspace,name= clustername)
        print("Find the existing cluster")
    except ComputeTargetException:
        print("Cluster not find - Creating cluster")
        is_new_cluster = True
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                            max_nodes=4)
        aml_compute = ComputeTarget.create(workspace, clustername, compute_config)

    aml_compute.wait_for_completion(show_output=True)
    
    return aml_compute


def get_environment():
    # create a new runconfig object
    run_config = RunConfiguration()

    # set Docker base image to the default CPU-based image
    run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE

    # use conda_dependencies.yml to create a conda environment in the Docker image for execution
    run_config.environment.python.user_managed_dependencies = False

    # specify CondaDependencies obj
    run_config.environment.python.conda_dependencies = CondaDependencies.create(
        conda_packages=['pandas','numpy'],
        pip_packages=['scikit-learn','joblib','azureml-sdk'],
        pin_sdk_version=False)

    return run_config

def main():
    
    print("Azure ML SDK Version: ", azureml.core.VERSION)
    print(sklearn.__version__)

    # Get workspace 

    ws = Workspace.from_config()
    print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')


    # Create AML compute
    aml_compute = get_aml_compute(workspace=ws)

    # Load and register the raw data 
    ## TODO We will need to change this as a parameter for new data
    cur_dir = os.getcwd()
    parent_dir = os.path.dirname(cur_dir)
    src_dir = os.path.join(cur_dir,'data')

    print(src_dir)

    data_store = ws.get_default_datastore()
    data_store.upload(src_dir=src_dir,target_path='cancer_data',overwrite=True,show_progress=True)
    
    ds_raw = Dataset.Tabular.from_delimited_files(path=data_store.path('cancer_data/cancer_data.csv'))
    ds_raw.register(workspace=ws,name='raw_data')

    # Create PipelineData

    ds_raw = ds_raw.as_named_input('raw_data')
    train_data = PipelineData("train_cancer_data",datastore=data_store).as_dataset()
    test_data = PipelineData("test_cancer_data",datastore=data_store).as_dataset()
    model_file = PipelineData("model_file",datastore=data_store)


    # Create Python Envirronment 
    run_config = get_environment()


    # 
    source_directory ='./scripts'
    step1 = PythonScriptStep(name="prepare_step",
                            script_name="prepare.py", 
                            arguments=["--input_data",ds_raw,"--train",train_data,"--test",test_data],
                            inputs=[ds_raw],
                            outputs=[train_data,test_data],
                            compute_target=aml_compute, 
                            runconfig=run_config,
                            source_directory=source_directory,
                            allow_reuse=True)
    print("Step Prepare created")

    step2 = PythonScriptStep(name="train_step",
                         script_name="train2.py", 
                         arguments=["--train",train_data,"--test",test_data,"--model_file",model_file],
                         inputs=[train_data,test_data],
                         outputs=[model_file],
                         compute_target=aml_compute, 
                         runconfig=run_config,
                         source_directory=source_directory,
                         allow_reuse=True)
    print("Step Train created")

    step3 = PythonScriptStep(name="register_step",
                         script_name="register.py", 
                         arguments=["--model_file",model_file],
                         inputs=[model_file],
                         compute_target=aml_compute, 
                         runconfig=run_config,
                         source_directory=source_directory,
                         allow_reuse=True)
    print("Step Register created")

    steps = [step1,step2,step3]
    pipeline1 = Pipeline(workspace=ws,steps=steps)
    
    run_exp = Experiment(workspace=ws, name="RF-BreastCancer-Pipeline")

    run_exp.submit(pipeline1,regenerate_ouputs=False)



if __name__ == '__main__':
    main()
