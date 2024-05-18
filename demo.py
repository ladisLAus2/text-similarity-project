from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
import sys
from text_similarity.configuration.gcloud_syncer import GCloudSyncher

syncher = GCloudSyncher()

syncher.syncronize_from_cloud_to_folder("text-similarity-project", 'data.zip','download/data.zip')