from azureml.core import  Workspace
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.environment import Environment 
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model,InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies

def main():

    ws = Workspace.from_config()
    print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

    model = Model(workspace=ws,name='breast-cancer',version=8)

    environment = Environment.from_conda_specification(name='mlopspython_scoring', file_path='./conda_dependencies_scoring.yml')
    environment.docker.enabled = True
    environment.docker.base_image = DEFAULT_CPU_IMAGE
    # environment = Environment('mlopspython_scoring')
    # environment.python.conda_dependencies = CondaDependencies.create(
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

    service_name = 'breast-cancer-custom-service'

    inference_config = InferenceConfig(entry_script='./scoring/score.py', environment=environment)
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1,
                                                    memory_gb=1 )
                                                    #vnet_name='csels-cdh-dev-vnet', 
                                                    #subnet_name='cdh-azml-dev-snet')



    service = Model.deploy(workspace=ws,
                        name=service_name,
                        models=[model],
                        inference_config=inference_config,
                        deployment_config=aci_config,
                        overwrite=True)
    
    service.wait_for_deployment(show_output=True)
    print("Service state:", service.state)

if __name__ == '__main__':
    main()