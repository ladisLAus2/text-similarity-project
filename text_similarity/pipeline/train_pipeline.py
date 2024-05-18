import sys
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.components.data_ingestion import DataIngestion
from text_similarity.entity.config_entity import (DataIngestionConfig)
from text_similarity.entity.artifact_entity import (DataIngestionArtifacts)

class TrainingPipeLine:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    
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
        
    
    def run_pipeline(self):
        logging.info('entered the run_pipeline method from TrainingPipeLine')
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info('exited the run_pipeline method from TrainingPipeLine')
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
            