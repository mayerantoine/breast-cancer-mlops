from azureml.core import  Workspace
from azureml.core.environment import Environment 
from azureml.core.webservice import AksWebservice
from azureml.core.model import Model,InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AksCompute

def main():
    ws = Workspace.from_config()
    
    service_name = 'breast-cancer-custom-service-aks'

    print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

    # TODO If model do not exist in registry check
    model = Model(workspace=ws,name='breast-cancer',version=8)

    # TODO Create yaml File and preferably Docker
    environment = Environment.from_conda_specification(name='mlopspython_scoring', file_path='./conda_dependencies_scoring.yml')
    environment.docker.enabled = True
    #environment = Environment('mlopspython_scoring')
    #environment.python.conda_dependencies = CondaDependencies.create(
    #            python_version='3.8',
    #            conda_packages=[
    #            'pip==20.2.4'],
    #            pip_packages=[
    #            'azureml-defaults',
    #            'pandas',
    #            'inference-schema[numpy-support]',
    #            'joblib',
    #            'numpy',
    #            'scikit-learn'
    #        ])


    inference_config = InferenceConfig(entry_script='./scoring/score.py', environment=environment)
    # TODO Check if compute exits
    aks_target = AksCompute(ws,"aks-cdh-dev")
    # If deploying to a cluster configured for dev/test, ensure that it was created with enough
    # cores and memory to handle this deployment configuration. Note that memory is also used by
    # things such as dependencies and AML components.
    deployment_config = AksWebservice.deploy_configuration(cpu_cores = 1, 
                                                        memory_gb = 1)
    aks_service = Model.deploy(ws, 
                            service_name, 
                            [model], 
                            inference_config, 
                            deployment_config, 
                            aks_target)
    aks_service.wait_for_deployment(show_output = True)
    print("Service state: ", aks_service.state)
    # print(aks_service.get_logs())
    

if __name__ == '__main__':
    main()