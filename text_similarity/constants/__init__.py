import os

from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
BUCKET_NAME = 'text-similarity-project'
ARTIFACTS_DIRECTORY = os.path.join('artifacts', TIMESTAMP)
DATASET_ARCHIEVE_NAME = 'data.zip'
PREMISE = 'premise'
HYPOTHESIS = 'hypothesis'
LABEL = 'label'



INGESTION_ARTIFACTS_DIRECTORY = 'IngestionArtifacts'
INGESTION_DATASET_1 = 'rec.csv'
INGESTION_DATASET_2 = 'mult.csv'
