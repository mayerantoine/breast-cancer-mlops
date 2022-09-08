import numpy as np
import json
from azureml.core import  Workspace
from azureml.core.webservice import AciWebservice,AksWebservice


def main():
    service_type = 'ACI'
    service_name = 'breast-cancer-custom-service'

    aml_workspace = Workspace.from_config()
    print('Workspace name: ' + aml_workspace.name, 
      'Azure region: ' + aml_workspace.location, 
      'Subscription id: ' + aml_workspace.subscription_id, 
      'Resource group: ' + aml_workspace.resource_group, sep = '\n')
    
    print("Fetching service")
    headers = {}
    if service_type == "ACI":
        service = AciWebservice(aml_workspace, service_name)
        service_name = 'breast-cancer-custom-service'
        print("Load ACI service")
    else:
        service = AksWebservice(aml_workspace, service_name)
        service_name = 'breast-cancer-custom-service-aks'
        print("Load AKS service")
    if service.auth_enabled:
        service_keys = service.get_keys()
        headers['Authorization'] = 'Bearer ' + service_keys[0]
    print("Testing service")
    print(f".url: {service.scoring_uri}")

    input_sample = np.array([[0.393845668409139, -0.7457496952627328, -0.22189398515428446, -0.7676382304726264, -0.6999246324662508, -0.17745016145311038, 
    -0.81484548808207, -0.770581761525873, -0.7189314051409194, 0.07367558407930598, -0.4665409990624931, -0.06425072325003647, -0.667435753669589, 
    -0.10209930505592255, -0.2928749021935234, -0.18393920631817678, -0.8204798568998558, -0.6065568523529427, -0.395651155530143, 0.3134950272756633,
    -0.8678658791041564, -0.6897420117050609, -0.5598110994362666, -0.723009965205552, -0.6542518616646612, -0.5936861839456196,
    -0.9574791012332506, -0.9270503937797329, -0.7552651134179409, -0.2651797994882257, -1.0640128515011344]])

    input_payload = json.dumps({
        'data': input_sample.tolist()
    })

    output = service.run(input_payload)

    print(output) 


if __name__ == '__main__':
    main()
