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
    
    def split_data(self, path):
        try:
            logging.info('entered split_data in ModelTrainer class')
            dataset = datasets.load_dataset('csv', data_files=path)
            split_dataset = dataset['train'].train_test_split(test_size=self.model_trainer_config.SPLIT_RATIO)
            train_dataset = split_dataset['train']
            test_dataset = split_dataset['test']
            logging.info('exited split_data from ModelTrainer class')
            return train_dataset, test_dataset
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
        
    def initialize_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info('entered initialize_model_trainer in ModelTrainer class')
            print(self.data_transformation_artifacts.transformed_data_path)
            train_dataset, test_dataset = self.split_data(path=self.data_transformation_artifacts.transformed_data_path)
            logging.info(f'train_dataset contains {len(train_dataset)} rows')
            logging.info(f'test_dataset contains {len(test_dataset)} rows')
            
            model = SBERT(model_name='bert-base-uncased', batch_size=self.model_trainer_config.BATCH_SIZE, learning_rate=self.model_trainer_config.LEARNING_RATE, scale=self.model_trainer_config.SCALE, epochs=self.model_trainer_config.EPOCH)
            
            logging.info('entering model training')
            model.train(train_dataset)
            logging.info('model training finished')
            
            logging.info('saving the model')
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIRECTORY, exist_ok=True)
            model.save(path=self.model_trainer_config.TRAINED_MODEL_PATH)
            
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH, 
                test_dataset=test_dataset
            )
            
            logging.info('exited from initialize_model_trainer in ModelTrainer class')
            logging.info('returning model_trainer_artifacts')
            return model_trainer_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
