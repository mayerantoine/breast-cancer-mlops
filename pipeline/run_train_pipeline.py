
from azureml.pipeline.core import PublishedPipeline
from azureml.core import Experiment, Workspace
from util.en_variables import Env 


def main():
  
  e = Env()
  pipelinename = e.pipeline_name
  experiment_name = e.experiment_pipeline_name
  p_version = e.model_version
  ws = Workspace.get(name= e.workspace_name,
                  subscription_id=e.subscription_id,
                  resource_group=e.resource_group,)
  
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
      if p.version == p_version:
        matched_pipes.append(p)
  # print(pipes)
  print(matched_pipes)
  # TODO we need to use the Pipeline Endpoint
  if len(matched_pipes) == 1 :
    published_pipeline = matched_pipes[0]
    print("published pipeline id is", published_pipeline.id)
    run_exp = Experiment(workspace=ws, name= experiment_name)
    run_exp.submit(published_pipeline,regenerate_ouputs=False)
  elif len(matched_pipes) == 0:
    published_pipeline = None
    raise KeyError("Unable to find a published pipeline for this build")

if __name__ == '__main__':
    main()
