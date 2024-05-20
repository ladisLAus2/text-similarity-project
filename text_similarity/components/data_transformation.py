import os
import torch
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split
from random import randint
import sys

from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.entity.config_entity import DataTransformationConfig
from text_similarity.entity.artifact_entity import DataIngestionArtifacts,DataTransformationArtifacts

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def data_preprocessing(self):
        try:
            logging.info('entered into the data_preprocessing method from DataTransformation class')
            
            dataset_1 = datasets.load_dataset('csv', data_files=self.data_ingestion_artifacts.dataset_1_file_path)
            dataset_2 = datasets.load_dataset('csv', data_files=self.data_ingestion_artifacts.dataset_2_file_path)
            validation_dataset = datasets.load_dataset('csv', data_files=self.data_ingestion_artifacts.validation_file_path)
            
            columns_to_remove_1 = [col for col in dataset_1['train'].column_names if col not in self.data_transformation_config.COLUMNS]
            columns_to_remove_2 = [col for col in dataset_2['train'].column_names if col not in self.data_transformation_config.COLUMNS]
            
            dataset_1 = dataset_1.remove_columns(columns_to_remove_1)
            dataset_2 = dataset_2.remove_columns(columns_to_remove_2)
            
            dataset = datasets.concatenate_datasets([dataset_1['train'], dataset_2['train']])
            dataset = dataset.select(range(1000))
            dataset = dataset.filter(
                lambda x: False if x['label'] != 0 else True
            )
            
            dataset = dataset.filter(
                lambda x: 0 if x['label'] in [1,2] else 1
            )
            dataset = dataset.filter(
                lambda x: isinstance(x['hypothesis'], str)
            )
            logging.info('exited into the data_preprocessing method from DataTransformation class')
            
            return dataset, validation_dataset
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info('entered into the initiate_data_transformation method from DataTransformation class')
            
            train_dataset, validation_dataset = self.data_preprocessing()
            df = pd.DataFrame(train_dataset)
            os.makedirs(self.data_transformation_config.TRANSFORMATION_ARTIFACTS_DIRECTORY, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFROMED_FILE_PATH, index=False, header=True)
            validation_dataset['train'].to_csv(self.data_transformation_config.VALIDATION_DATASET_PATH)
            data_transformation_artifacts = DataTransformationArtifacts(transformed_data_path=self.data_transformation_config.TRANSFROMED_FILE_PATH, validation_dataset=self.data_transformation_config.VALIDATION_DATASET_PATH)
            
            logging.info('exited into the initiate_data_transformation method from DataTransformation class')
            return data_transformation_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e