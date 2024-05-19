from dataclasses import dataclass
import datasets

@dataclass 
class DataIngestionArtifacts:
    dataset_1_file_path: str
    dataset_2_file_path: str
    
@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str
    
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    test_dataset: datasets.arrow_dataset.Dataset