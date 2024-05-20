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
INGESTION_DATASET_3 = 'validation.csv'



TRANSFORMATION_ARTIFACTS_DIRECTORY = 'TransformationArtifacts'
TRANSFORMED_FILE_NAME = 'filtered_dataset.csv'
DATA_DIRECTORY = 'data'
COLUMNS = [PREMISE, HYPOTHESIS, LABEL]



MODEL_TRAINER_ARTIFACTS_DIRECTORY = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIRECTORY = 'trained_model'
TRAINED_MODEL_NAME = 'model_mnr'
TRAIN_DATASET = 'train_dataset.csv'
TEST_DATASET = 'test_dataset.csv'

EPOCH = 1
BATCH_SIZE = 32
SPLIT_RATIO = 0.1
LEARNING_RATE = 2e-5

BASE_MODEL = 'roberta-base'
SCALE = 20.0


MODEL_EVALUATION_ARTIFACTS_DIRECTORY = 'ModelEvaluationArtifacts'
BEST_MODEL_DIRECTORY = 'best_model'
MODEL_EVALUATION_LOSS_FILE = 'loss.csv'
EVALUATION_DATSET = 'validation.csv'

APP_HOST = '0.0.0.0'
APP_PORT = 8080

