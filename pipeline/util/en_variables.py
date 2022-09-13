"""Env dataclass to load and hold all environment variables
"""
from dataclasses import dataclass
import os
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Env:
    """Loads all environment variables into a predefined set of properties
    """

    # to load .env file into environment variables for local execution
    load_dotenv()
    workspace_name: Optional[str] = os.environ.get("WORKSPACE_NAME")
    resource_group: Optional[str] = os.environ.get("RESOURCE_GROUP")
    subscription_id: Optional[str] = os.environ.get("SUBSCRIPTION_ID")
    tenant_id: Optional[str] = os.environ.get("TENANT_ID")
    app_id: Optional[str] = os.environ.get("SP_APP_ID")
    app_secret: Optional[str] = os.environ.get("SP_APP_SECRET")
    vm_size: Optional[str] = os.environ.get("AML_COMPUTE_CLUSTER_CPU_SKU")
    compute_name: Optional[str] = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    vm_priority: Optional[str] = os.environ.get(
        "AML_CLUSTER_PRIORITY", "lowpriority"
    )  # NOQA: E501
    min_nodes: int = int(os.environ.get("AML_CLUSTER_MIN_NODES", 0))
    max_nodes: int = int(os.environ.get("AML_CLUSTER_MAX_NODES", 4))
    v_net: Optional[str] = os.environ.get("V_NET")
    sub_net: Optional[str] = os.environ.get("SUB_NET")
    
    
    build_id: Optional[str] = os.environ.get("BUILD_BUILDID")
    pipeline_name: Optional[str] = os.environ.get("TRAINING_PIPELINE_NAME")
    sources_directory_train: Optional[str] = os.environ.get(
        "SOURCES_DIR_TRAIN"
    )  # NOQA: E501
    prepare_script_path: Optional[str] = os.environ.get("PREPARE_SCRIPT_PATH")
    train_script_path: Optional[str] = os.environ.get("TRAIN_STEP_SCRIPT_PATH")
    evaluate_script_path: Optional[str] = os.environ.get(
        "EVALUATE_SCRIPT_PATH"
    )  # NOQA: E501
    register_script_path: Optional[str] = os.environ.get(
        "REGISTER_SCRIPT_PATH"
    )  # NOQA: E501
    model_file_name: Optional[str] = os.environ.get("MODEL_FILE_NAME")
    register_model_name: Optional[str] = os.environ.get("REGISTER_MODEL_NAME")
    experiment_name: Optional[str] = os.environ.get("EXPERIMENT_NAME")
    experiment_pipeline_name: Optional[str] = os.environ.get("EXPERIMENT_PIPELINE_NAME")
    model_version: Optional[str] = os.environ.get("MODEL_VERSION")
    image_name: Optional[str] = os.environ.get("IMAGE_NAME")
    db_cluster_id: Optional[str] = os.environ.get("DB_CLUSTER_ID")
    score_script: Optional[str] = os.environ.get("SCORE_SCRIPT")
    build_uri: Optional[str] = os.environ.get("BUILD_URI")
    dataset_name: Optional[str] = os.environ.get("DATASET_NAME")
    datastore_name: Optional[str] = os.environ.get("DATASTORE_NAME")
    data_file_name: Optional[str] = os.environ.get("DATA_FILE_NAME")
    data_src_folder: Optional[str] = os.environ.get("DATA_SRC_FOLDER")

    dataset_version: Optional[str] = os.environ.get("DATASET_VERSION")
    run_evaluation: Optional[str] = os.environ.get("RUN_EVALUATION", "true")
    allow_run_cancel: Optional[str] = os.environ.get(
        "ALLOW_RUN_CANCEL", "true"
    )  # NOQA: E501
    aml_env_name: Optional[str] = os.environ.get("AML_ENV_NAME")
    aml_env_train_conda_dep_file: Optional[str] = os.environ.get(
        "AML_ENV_TRAIN_CONDA_DEP_FILE", "conda_dependencies.yml"
    )

    rebuild_env: Optional[bool] = os.environ.get(
        "AML_REBUILD_ENVIRONMENT", "false"
    ).lower().strip() == "true"

    use_gpu_for_scoring: Optional[bool] = os.environ.get(
        "USE_GPU_FOR_SCORING", "false"
    ).lower().strip() == "true"
    aml_env_score_conda_dep_file: Optional[str] = os.environ.get(
        "AML_ENV_SCORE_CONDA_DEP_FILE", "conda_dependencies_scoring.yml"
    )
    aml_env_scorecopy_conda_dep_file: Optional[str] = os.environ.get(
        "AML_ENV_SCORECOPY_CONDA_DEP_FILE", "conda_dependencies_scorecopy.yml"
    )
    vm_size_scoring: Optional[str] = os.environ.get(
        "AML_COMPUTE_CLUSTER_CPU_SKU_SCORING"
    )
    compute_name_scoring: Optional[str] = os.environ.get(
        "AML_COMPUTE_CLUSTER_NAME_SCORING"
    )
    vm_priority_scoring: Optional[str] = os.environ.get(
        "AML_CLUSTER_PRIORITY_SCORING", "lowpriority"
    )
    min_nodes_scoring: int = int(
        os.environ.get("AML_CLUSTER_MIN_NODES_SCORING", 0)
    )  # NOQA: E501
    max_nodes_scoring: int = int(
        os.environ.get("AML_CLUSTER_MAX_NODES_SCORING", 4)
    )  # NOQA: E501
    rebuild_env_scoring: Optional[bool] = os.environ.get(
        "AML_REBUILD_ENVIRONMENT_SCORING", "false"
    ).lower().strip() == "true"
    aml_env_name_scoring: Optional[str] = os.environ.get(
        "AML_ENV_NAME_SCORING"
    )  # NOQA: E501
    aml_env_name_score_copy: Optional[str] = os.environ.get(
        "AML_ENV_NAME_SCORE_COPY"
    )  # NOQA: E501

    aci_service_name: Optional[str] = os.environ.get("ACI_SERVICE_NAME")
    aks_service_name: Optional[str] = os.environ.get("AKS_SERVICE_NAME")
    aci_cpu_core: int = int(os.environ.get("ACI_CPU_CORE"))
    aci_memory_gb: int = int(os.environ.get("ACI_MEMORY_GB"))
