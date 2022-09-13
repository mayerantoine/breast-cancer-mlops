import os
import azureml
import sklearn
from azureml.core import  Workspace,RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Dataset
from util.en_variables import Env
from util.aml_service import get_aml_compute, get_environment


def main():
    """
    Build and submit pipeline
    """

    print("Azure ML SDK Version: ", azureml.core.VERSION)
    print("sklearn version: ",sklearn.__version__)

    # Get workspace
    e = Env()
    workspace = Workspace.get(name= e.workspace_name,
                    subscription_id=e.subscription_id,
                    resource_group=e.resource_group,)

    # ws = Workspace.from_config()
    print('Workspace name: ' + workspace.name,
      'Azure region: ' + workspace.location,
      'Subscription id: ' + workspace.subscription_id,
      'Resource group: ' + workspace.resource_group, sep = '\n')

    # Create AML compute
    aml_compute = get_aml_compute(workspace=workspace,
                                clustername=e.compute_name,
                                vm_size=e.vm_size,
                                resource_grp=e.resource_group,
                                vnet_name=e.v_net,
                                subnet_name=e.sub_net,
                                max_nodes=e.max_nodes)


    # Load and register the raw data
    ## TODO We will need to change this as a parameter for new data
    cur_dir = os.getcwd()
    print("working directory", cur_dir)

    src_dir = os.path.join(cur_dir,e.data_src_folder)
    print("data source dir:",src_dir)
    file_name = e.data_file_name
    target_path='cancer_data'


    data_store = workspace.get_default_datastore()
    data_store.upload(src_dir=src_dir,
                                target_path=target_path,
                                overwrite=True,
                                show_progress=True)
    ds_raw = Dataset.Tabular.from_delimited_files(path=data_store.path(f'{target_path}/{file_name}'))

    #ds_raw = Dataset.Tabular.from_delimited_files(src_dir=src_dir,
    #       target=DataPath(data_store,  target_path),
    #       overwrite=True,
    #       show_progress=True)

    ds_raw.register(workspace=workspace,
                    name=e.dataset_name,
                    description="diabetes training data",
                    tags={"format": "CSV"},
                    create_new_version=True)

    # Create PipelineData
    ds_raw = ds_raw.as_named_input('raw_data')
    train_data = PipelineData("train_cancer_data",datastore=data_store).as_dataset()
    test_data = PipelineData("test_cancer_data",datastore=data_store).as_dataset()
    model_file = PipelineData("model_file",datastore=data_store)

    # Create Python Envirronment
    file_env_path = os.path.join(cur_dir,e.aml_env_train_conda_dep_file)
    print("environment file path:",file_env_path)
    environment  = get_environment(workspace = workspace,
                                    env_name=e.aml_env_name,
                                    env_file_path=file_env_path)
    run_config = RunConfiguration()
    run_config.environment = environment

    #run_config.environment.python.conda_dependencies = CondaDependencies.create(
    #    python_version='3.8',
    #    conda_packages=['pandas','numpy'],
    #    pip_packages=['azureml-defaults','scikit-learn','joblib','azureml-sdk'],
    #    pin_sdk_version=False)

    # Prepare step
    source_directory = e.sources_directory_train
    step1 = PythonScriptStep(name="prepare_step",
                            script_name='prepare.py',
                            arguments=["--input_data",ds_raw,"--train",train_data,"--test",test_data],
                            inputs=[ds_raw],
                            outputs=[train_data,test_data],
                            compute_target = aml_compute,
                            runconfig=run_config,
                            source_directory=source_directory,
                            allow_reuse=False)
    print("Step Prepare created")

    step2 = PythonScriptStep(name="train_step",
                         script_name=e.train_script_path,
                         arguments=["--train",train_data,"--test",test_data,"--model_file",model_file],
                         inputs=[train_data,test_data],
                         outputs=[model_file],
                         compute_target=aml_compute,
                         runconfig=run_config,
                         source_directory=source_directory,
                         allow_reuse=True)
    print("Step Train created")

    step3 = PythonScriptStep(name="register_step",
                         script_name=e.register_script_path,
                         arguments=["--model_file",model_file],
                         inputs=[model_file],
                         compute_target=aml_compute,
                         runconfig=run_config,
                         source_directory=source_directory,
                         allow_reuse=True)
    print("Step Register created")

    steps = [step1,step2,step3]
    train_pipeline = Pipeline(workspace=workspace,steps=steps)

    print("Validating pipeline...")
    train_pipeline.validate()

    pipelinename = e.pipeline_name

    print("Publishing pipeline...")
    published_pipeline = train_pipeline.publish(
        name=pipelinename,
        description="Model training/retraining pipeline",
        version=e.model_version )

    print(f"Published pipeline: {published_pipeline.name}")
    print(f"for build {published_pipeline.version}")


if __name__ == '__main__':
    main()
