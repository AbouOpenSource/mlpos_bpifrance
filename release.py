import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient()
# Parametrizing the right experiment path using widgets
mlflow.set_tracking_uri('http://localhost:5000')
experiment_name = 'Wine Regression'
experiment = client.get_experiment_by_name(experiment_name)
experiment_ids = [experiment.experiment_id]
print("Experiment IDs:", experiment_ids)

# Setting the decision criteria for a best run
query = "metrics.mae > 0.2"
runs = client.search_runs(experiment_ids, query, ViewType.ALL)

# Searching throught filtered runs to identify the best_run and build the model URI to programmatically reference later
accuracy_high = None
best_run = None
for run in runs:
  if (accuracy_high == None or run.data.metrics['mae'] > accuracy_high):
    accuracy_high = run.data.metrics['mae']
    best_run = run
run_id = best_run.info.run_id
print('Highest Accuracy: ', accuracy_high)
print('Run ID: ', run_id)

model_uri = "runs:/" + run_id + "/model"



import time

# Check if model is already registered
model_name = "ElasticnetWineModel"
try:
  registered_model = client.get_registered_model(model_name)
except:
  registered_model = client.create_registered_model(model_name)

# Create the model source
model_source = f"{best_run.info.artifact_uri}/model"
print(model_source)

# Archive old production model
max_version = 0
for mv in client.search_model_versions("name='ElasticnetWineModel'"):
  current_version = int(dict(mv)['version'])
  if current_version > max_version:
    max_version = current_version
  if dict(mv)['current_stage'] == 'Production':
    version = dict(mv)['version']
    client.transition_model_version_stage(model_name, version, stage='Archived')

# Create a new version for this model with best metric (accuracy)
client.create_model_version(model_name, model_source, run_id)
# Check the status of the created model version (it has to be READY)
status = None
while status != 'READY':
  for mv in client.search_model_versions(f"run_id='{run_id}'"):
    status = mv.status if int(mv.version)==max_version + 1 else status
  time.sleep(5)

# Promote the model version to production stage
client.transition_model_version_stage(model_name, max_version + 1, stage='Production')