import os

class GCloudSyncher:
    def syncronize_from_folder_to_cloud(self, bucket_url, filepath, filename):
        command = f"gsutil cp {filepath}/{filename} gs://{bucket_url}/"
        os.system(command)
    
    def syncronize_from_cloud_to_folder(self, bucket_url, filename, destination):
        command = f"gsutil cp gs://{bucket_url}/{filename} {destination}/{filename}"
        os.system(command)