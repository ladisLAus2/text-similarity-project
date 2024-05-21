import os
from google.cloud import storage
from tqdm import tqdm
import hashlib
import base64
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

class GCloudSyncher:
    def __init__(self):
        self.client = storage.Client()
            
    def download_file_from_cloud(self, blob_name, file_path, bucket_name):
        try:
            print(f'downloading file {blob_name} from cloud')
            # bucket = self.client.get_bucket(bucket_name)
            # blob = bucket.blob(blob_name)
            # with open(file_path, 'wb') as f:
            #     self.client.download_blob_to_file(blob, f)
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            with open(file_path, 'wb') as f:
                with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                    
                    self.client.download_blob_to_file(blob, file_obj)
            return True
        except Exception as e:
            print(e)
            return False
        
    def upload_to_cloud(self, blob_name, file_path, bucket_name):
        try:
            print(f'uploading file {blob_name} to cloud')
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            # blob.upload_from_filename(file_path)
            with open(file_path, "rb") as in_file:
                total_bytes = os.fstat(in_file.fileno()).st_size
                with tqdm.wrapattr(in_file, "read", total=total_bytes, miniters=1, desc="upload to %s" % bucket_name) as file_obj:
                    blob.upload_from_file(
                        file_obj,
                        size=total_bytes,
                    )
            
            return True
        except Exception as e:
            print(e)
            return False
        
    def generate_md5(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return base64.b64encode(hash_md5.digest()).decode('utf-8')
    
    def get_gcs_file_md5(self, bucket_name, file_name):
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.get_blob(file_name)
        return blob.md5_hash