import sys
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.components.data_ingestion import DataIngestion
from text_similarity.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig)
from text_similarity.entity.artifact_entity import (DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts)
from text_similarity.components.data_transformation import DataTransformation
from text_similarity.components.model_trainer import ModelTrainer
from text_similarity.components.model_evaluation import ModelEvaluation


class TrainingPipeLine:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        
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
    
    
    def start_model_trainer(self, data_transformation_artifacts) -> ModelTrainerArtifacts:
        try:
            logging.info('entered the start_model_trainer method from TrainingPipeLine class')
            
            model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts,
                                        model_trainer_config=self.model_trainer_config)
            model_trainer_artifacts = model_trainer.initialize_model_trainer()
            
            logging.info('exited the start_model_trainer method from TrainingPipeLine class')
            return model_trainer_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
    
    def start_model_evaluation(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, data_transformatio_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig) -> ModelEvaluationArtifacts:
        try:
            logging.info('entered the start_model_evaluation method from TrainingPipeLine class')
            
            model_evaluation = ModelEvaluation(model_evaluation_config, model_trainer_artifacts, data_transformatio_artifacts, model_trainer_config)
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info('exited the start_model_evaluation method from TrainingPipeLine class')
            return model_evaluation_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
    
    def run_pipeline(self):
        try:
            logging.info('entered the run_pipeline method from TrainingPipeLine')
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts)
            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts)
            model_evaluation_artifacts = self.start_model_evaluation(model_evaluation_config=self.model_evaluation_config,
                                                                    model_trainer_artifacts=model_trainer_artifacts,
                                                                    data_transformatio_artifacts=data_transformation_artifacts,
                                                                    model_trainer_config=self.model_trainer_config)
            logging.info('exited the run_pipeline method from TrainingPipeLine')
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
            