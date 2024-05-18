import os
import sys
from zipfile import ZipFile
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.configuration.gcloud_syncer import GCloudSyncher
from text_similarity.entity.config_entity import DataIngestionConfig
from text_similarity.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.cloud = GCloudSyncher()
    
    def get_data_from_cloud(self) -> None:
        try:
            logging.info('entered the get_data_from_cloud method from DataIngestion class')
            os.makedirs(self.data_ingestion_config.INGESTION_ARTIFACTS_DIRECTORY, exist_ok=True)
            self.cloud.syncronize_from_cloud_to_folder(self.data_ingestion_config.BUCKET_NAME, self.data_ingestion_config.ARCHIEVE_NAME, self.data_ingestion_config.INGESTION_ARTIFACTS_DIRECTORY)
            logging.info('exited the get_data_from_cloud method from DataIngestion class')
            
        except Exception as e:
            raise ExceptionHandler(e, sys) from e   

    def unzip_data(self):
        logging.info('entered the unzip method from DataIngestion class')
        try:
            with ZipFile(self.data_ingestion_config.ARCHIEVE_FILE_PATH,'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ARCHIVE_DIRECTORY)
            
            logging.info('exited the unzip method of Dataingestion class')
            return self.data_ingestion_config.ARTIFACTS_DIRECTORY, self.data_ingestion_config.NEW_ARTIFACTS_DIRECTORY
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
    
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info('entered the initiate_data_ingestion method from DataIngestion class')
        try:
            self.get_data_from_cloud()
            logging.info('downloaded the data from GCloud bucket')
            dataset_1_file_path, dataset_2_file_path = self.unzip_data()
            logging.info('unzipped the data')

            data_ingestion_artifacts = DataIngestionArtifacts(
                dataset_1_file_path=dataset_1_file_path,
                dataset_2_file_path=dataset_2_file_path
            )
            logging.info('exited the initiate_data_ingestion method of DataIngestion class')
            return data_ingestion_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e