import os
from google.cloud import storage
from tqdm import tqdm

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

class GCloudSyncher:
    def __init__(self):
        self.client = storage.Client()
            
    def download_file_from_cloud(self, blob_name, file_path, bucket_name):
        try:
            print(f'downloading file {blob_name} from cloud')
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            with open(file_path, 'wb') as f:
                self.client.download_blob_to_file(blob, f)
            return True
        except Exception as e:
            print(e)
            return False
        
    def upload_to_cloud(self, blob_name, file_path, bucket_name):
        try:
            print(f'uploading file {blob_name} to cloud')
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            return True
        except Exception as e:
            print(e)
            return False