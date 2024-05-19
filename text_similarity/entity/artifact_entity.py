from dataclasses import dataclass

@dataclass 
class DataIngestionArtifacts:
    dataset_1_file_path: str
    dataset_2_file_path: str
    
@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str