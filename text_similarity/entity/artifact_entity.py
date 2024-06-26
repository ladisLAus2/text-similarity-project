from dataclasses import dataclass
import datasets

@dataclass 
class DataIngestionArtifacts:
    dataset_1_file_path: str
    dataset_2_file_path: str
    validation_file_path: str
    
@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str
    validation_dataset: str
    
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    validation_dataset: str
    
@dataclass 
class ModelEvaluationArtifacts:
    is_model_accepted: bool
    
@dataclass
class ModelPusherArtifacts:
    bucket_name: str