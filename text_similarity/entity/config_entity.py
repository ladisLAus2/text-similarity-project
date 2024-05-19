from dataclasses import dataclass
from text_similarity.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.ARCHIEVE_NAME = DATASET_ARCHIEVE_NAME
        self.INGESTION_ARTIFACTS_DIRECTORY : str = os.path.join(os.getcwd(), ARTIFACTS_DIRECTORY, INGESTION_ARTIFACTS_DIRECTORY)
        self.ARTIFACTS_DIRECTORY: str = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY, INGESTION_DATASET_1)
        self.NEW_ARTIFACTS_DIRECTORY: str = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY, INGESTION_DATASET_2)
        self.ARCHIVE_DIRECTORY = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY)
        self.ARCHIEVE_FILE_PATH = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY, self.ARCHIEVE_NAME)
        

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.TRANSFORMATION_ARTIFACTS_DIRECTORY: str = os.path.join(os.getcwd(), ARTIFACTS_DIRECTORY, TRANSFORMATION_ARTIFACTS_DIRECTORY)
        self.TRANSFROMED_FILE_PATH = os.path.join(self.TRANSFORMATION_ARTIFACTS_DIRECTORY, TRANSFORMED_FILE_NAME)
        self.COLUMNS = COLUMNS
        
        
@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIRECTORY : str = os.path.join(os.getcwd(), ARTIFACTS_DIRECTORY, MODEL_TRAINER_ARTIFACTS_DIRECTORY)
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIRECTORY, TRAINED_MODEL_NAME)
        self.TEST_DATASET = os.path.join(self.TRAINED_MODEL_DIRECTORY, TEST_DATASET)
        self.TRAIN_DATASET = os.path.join(self.TRAINED_MODEL_DIRECTORY, TRAIN_DATASET)
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.SPLIT_RATIO = SPLIT_RATIO
        self.BASE_MODEL = BASE_MODEL
        self.SCALE = SCALE
        self.LEARNING_RATE = LEARNING_RATE