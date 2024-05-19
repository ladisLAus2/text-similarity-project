import sys
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.components.data_ingestion import DataIngestion
from text_similarity.entity.config_entity import (DataIngestionConfig, DataTransformationConfig)
from text_similarity.entity.artifact_entity import (DataIngestionArtifacts, DataTransformationArtifacts)
from text_similarity.components.data_transformation import DataTransformation

class TrainingPipeLine:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info('entered the start_data_ingestion method from TrainingPipeLine class')
        try:
            logging.info('getting the data from GCloud bucket')
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info('got the data from GCloud')
            logging.info('exited the start_data_ingestion method from TrainingPipeLine class')
            return data_ingestion_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
    
    
    def start_data_transformation(self,data_ingestion_artifacts) -> DataTransformationArtifacts:
        try:
            logging.info('entered the start_data_transformation method from TrainingPipeLine class')
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config, data_ingestion_artifacts=data_ingestion_artifacts)
            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            logging.info('exited the start_data_transformation method from TrainingPipeLine class')
            return data_transformation_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
    
    
    def run_pipeline(self):
        logging.info('entered the run_pipeline method from TrainingPipeLine')
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts)
            logging.info('exited the run_pipeline method from TrainingPipeLine')
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
            