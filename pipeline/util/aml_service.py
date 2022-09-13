
from azureml.core import  Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE
from util.en_variables import Env

def get_aml_compute(workspace,clustername,vm_size,resource_grp, vnet_name,subnet_name,max_nodes):
    """
     Retreive or Create compute for training
    """
    clustername = 'StandardDS12CPU'
    try:
        aml_compute = ComputeTarget(workspace = workspace,name= clustername)
        print("Find the existing cluster")
    except ComputeTargetException:
        print("Cluster not find - Creating cluster.....")
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                vnet_name=subnet_name,
                                                                vnet_resourcegroup_name=resource_grp,
                                                                subnet_name=subnet_name,
                                                            max_nodes=max_nodes)
        aml_compute = ComputeTarget.create(workspace, clustername, compute_config)

    aml_compute.wait_for_completion(show_output=True)

    return aml_compute


def get_environment(workspace,
                    env_name,
                    env_file_path):
    """
    Create the environment for training and inference
    """

    e = Env()
    environments = Environment.list(workspace=workspace)
    restored_environment = None
    for env in environments:
        if env == e.aml_env_name:
            restored_environment = environments[env]

    if restored_environment is None:
        new_env = Environment.from_conda_specification(name=env_name,
                                            file_path=env_file_path)
        restored_environment = new_env
        #restored_environment.docker.enabled = True
        restored_environment.docker.base_image = DEFAULT_CPU_IMAGE
        restored_environment.register(workspace=workspace)
    else:
        print(restored_environment)

    return restored_environment
