
from azureml.pipeline.core import PublishedPipeline
from azureml.core import Experiment, Workspace
import argparse



def main():

    pipelinename = 'cancer-Training-Pipeline'
    experiment_name = "RF-BreastCancer-Pipeline"
    p_version = 3

    ws = Workspace.from_config()
    
    print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

    pipes = PublishedPipeline.list(ws)
    matched_pipes = []
 

    for p in pipes:
      #print(p.version)
      if p.name == pipelinename:
        #print(p)
        if p.version == '3':
          matched_pipes.append(p)
    # print(pipes)
    print(matched_pipes)

    # TODO we need to use the Pipeline Endpoint
    if len(matched_pipes) == 1 :
      published_pipeline = matched_pipes[0]
      run_exp = Experiment(workspace=ws, name= experiment_name)
      run_exp.submit(published_pipeline,regenerate_ouputs=False)

    

if __name__ == '__main__':
    main()
