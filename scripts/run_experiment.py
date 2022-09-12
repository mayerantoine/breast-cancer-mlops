from azureml.core import Workspace, Experiment, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig, Environment

def main():
    
    # get workspace
    ws = Workspace.from_config()
    print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

    # create experiment
    exp = Experiment(workspace=ws, name="RF-BreastCancer")
    
    # get compute
    clustername = 'StandardDS12CPU'
    try:
        aml_cluster = ComputeTarget(workspace = ws,name= clustername)
        print("Find the existing cluster")
    except ComputeTargetException:
        print("Cluster not find - Creating cluster.....")
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                                vnet_name='csels-cdh-dev-vnet',
                                                                vnet_resourcegroup_name='CSELS-CDH-DEV',
                                                                subnet_name='cdh-azml-dev-snet',
                                                            max_nodes=4)
        aml_cluster = ComputeTarget.create(ws, clustername, compute_config)

    aml_cluster.wait_for_completion(show_output=True)
    
    # get environment 
    sklearn_env = Environment.from_conda_specification(name='mlopspython', file_path='conda_dependencies.yml')
    sklearn_env.docker.enabled = True


    # get data
    ## TODO configuration depends on the workspace


    data_store = ws.get_default_datastore()
    data_store.upload(src_dir='./data',target_path='cancer_data',overwrite=True,show_progress=True)

    # create script config
    estimator = ScriptRunConfig(source_directory='./scripts',
                      script='train.py',
                      compute_target=aml_cluster,
                      environment=sklearn_env)

    # submit training
    # TODO: Submit your experiment
    print("Submit Experiment")
    run = exp.submit(estimator)
    #run.wait_for_completion(show_output=True, 
    #                        wait_post_processing=False, 
    #                        raise_on_error=True)

    print("Experiment submitted")
    

if __name__ =='__main__':
    main()