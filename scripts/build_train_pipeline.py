import os
import azureml
import sklearn
from azureml.core import  Workspace
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

def get_aml_compute(workspace):
    """
     Retreive or Create compute for training
    """
    clustername = 'StandardDS12CPU'
    try:
        aml_compute = ComputeTarget(workspace = workspace,name= clustername)
        print("Find the existing cluster")
    except ComputeTargetException:
        print("Cluster not find - Creating cluster.....")
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                                vnet_name='csels-cdh-dev-vnet',
                                                                vnet_resourcegroup_name='CSELS-CDH-DEV',
                                                                subnet_name='cdh-azml-dev-snet',
                                                            max_nodes=4)
        aml_compute = ComputeTarget.create(workspace, clustername, compute_config)

    aml_compute.wait_for_completion(show_output=True)
        
    return aml_compute


def get_environment():
    """
    Create the environment for training and inference
    """
    # create a new runconfig object
    run_config = RunConfiguration()

    # set Docker base image to the default CPU-based image
    run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE

    # use conda_dependencies.yml to create a conda environment in the Docker image for execution
    run_config.environment.python.user_managed_dependencies = False

    # specify CondaDependencies obj
    run_config.environment.python.conda_dependencies = CondaDependencies.create(
        python_version='3.8',
        conda_packages=['pandas','numpy','matplotlib'],
        pip_packages=['scikit-learn','joblib','azureml-sdk'],
        pin_sdk_version=False)

    return run_config


def main():
    """
    Build and submit pipeline
    """
    
    print("Azure ML SDK Version: ", azureml.core.VERSION)
    print("sklearn version: ",sklearn.__version__)

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

    # Prepare step
    source_directory ='./scripts'
    step1 = PythonScriptStep(name="prepare_step",
                            script_name="prepare.py", 
                            arguments=["--input_data",ds_raw,"--train",train_data,"--test",test_data],
                            inputs=[ds_raw],
                            outputs=[train_data,test_data],
                            compute_target = aml_compute, 
                            runconfig=run_config,
                            source_directory=source_directory,
                            allow_reuse=True)
    print("Step Prepare created")

    step2 = PythonScriptStep(name="train_step",
                         script_name="train_step.py", 
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
