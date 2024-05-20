import os
import sys
import pickle
import pandas as pd
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.constants import *
from sklearn.model_selection import train_test_split
import torch
import transformers
from random import randint
import pandas as pd
import datasets
from text_similarity.entity.config_entity import ModelTrainerConfig
from text_similarity.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from text_similarity.ml.model import SBERT

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, 
                model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
    
    def get_train_data(self, path):
        try:
            logging.info('entered split_data in ModelTrainer class')
            dataset = datasets.load_dataset('csv', data_files=path)
            
            logging.info('exited split_data from ModelTrainer class')
            return dataset['train']
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
        
    def initialize_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info('entered initialize_model_trainer in ModelTrainer class')
            
            dataset = self.get_train_data(path=self.data_transformation_artifacts.transformed_data_path)
            
            model = SBERT(model_name='bert-base-uncased', batch_size=self.model_trainer_config.BATCH_SIZE, learning_rate=self.model_trainer_config.LEARNING_RATE, scale=self.model_trainer_config.SCALE, epochs=self.model_trainer_config.EPOCH)
            
            logging.info('entering model training')
            model.train(dataset)
            logging.info('model training finished')
            
            logging.info('saving the model')
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIRECTORY, exist_ok=True)
            model.save(path=self.model_trainer_config.TRAINED_MODEL_PATH)
            dataset.to_csv(self.model_trainer_config.TRAIN_DATASET)
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH, 
                validation_dataset=self.data_transformation_artifacts.validation_dataset
            )
            
            logging.info('exited from initialize_model_trainer in ModelTrainer class')
            logging.info('returning model_trainer_artifacts')
            return model_trainer_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
