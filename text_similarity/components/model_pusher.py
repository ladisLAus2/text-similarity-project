import sys
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.configuration.gcloud_syncer import GCloudSyncher
from text_similarity.entity.config_entity import ModelPusherConfig
from text_similarity.entity.artifact_entity import ModelPusherArtifacts
import os
class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
        self.gcloud = GCloudSyncher()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        try:
            logging.info('entered initiate_model_pusher in ModelPusher class')
            
            # self.gcloud.syncronize_from_folder_to_cloud(self.model_pusher_config.BUCKET_NAME, self.model_pusher_config.TRAINED_MODEL_PATH, self.model_pusher_config.MODEL_NAME)
            print('uploading model to cloud')
            self.gcloud.upload_to_cloud(self.model_pusher_config.MODEL_NAME, os.path.join(self.model_pusher_config.TRAINED_MODEL_PATH, self.model_pusher_config.MODEL_NAME), self.model_pusher_config.BUCKET_NAME)
            self.gcloud.upload_to_cloud(self.model_pusher_config.CONFIG_NAME, os.path.join(self.model_pusher_config.TRAINED_MODEL_PATH, self.model_pusher_config.CONFIG_NAME), self.model_pusher_config.BUCKET_NAME)
            print('uploaded model to gcloud')
            logging.info('uploaded model to gcloud')
            
            model_pusher_artifact = ModelPusherArtifacts(bucket_name=self.model_pusher_config.BUCKET_NAME)
            
            logging.info('exited initiate_model_pusher in ModelPusher class')
            return model_pusher_artifact
        except Exception as e:
            raise ExceptionHandler(e, sys) from e