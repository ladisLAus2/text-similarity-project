import os
import sys
import numpy as np
import pandas as pd
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
import torch
from text_similarity.constants import *
from text_similarity.configuration.gcloud_syncer import GCloudSyncher
from text_similarity.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from text_similarity.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts
import transformers
import datasets
from scipy.stats import spearmanr
from sentence_transformers import util
from sklearn.metrics import accuracy_score


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, data_transformatio_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformatio_artifacts = data_transformatio_artifacts
        self.model_trainer_config = model_trainer_config
        self.gcloud = GCloudSyncher()

    def get_best_model_from_gcloud(self):
        try:
            logging.info('entered get_best_model_from_gcloud in ModelEvaluation class')
            os.makedirs(self.model_evaluation_config.BEST_MODEL_PATH, exist_ok=True)
            # self.gcloud.syncronize_from_cloud_to_folder(
            #     self.model_evaluation_config.BUCKET_NAME,
            #     self.model_evaluation_config.MODEL_NAME,
            #     self.model_evaluation_config.BEST_MODEL_PATH,flag=True
            # )
            print('donwloading model from cloud')
            result_model = self.gcloud.download_file_from_cloud(blob_name=self.model_evaluation_config.MODEL_NAME,
                                                 file_path=os.path.join(self.model_evaluation_config.BEST_MODEL_PATH, self.model_evaluation_config.MODEL_NAME),
                                                 bucket_name=self.model_evaluation_config.BUCKET_NAME)
            
            result_config = self.gcloud.download_file_from_cloud(blob_name=self.model_evaluation_config.TRAINED_MODEL_CONFIG,
                                                 file_path=os.path.join(self.model_evaluation_config.BEST_MODEL_PATH, self.model_evaluation_config.TRAINED_MODEL_CONFIG),
                                                 bucket_name=self.model_evaluation_config.BUCKET_NAME)
            if result_model and result_config:
                print('model and config were downloaded from cloud')
            else:
                print('model and config were not downloaded from cloud')
                
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_PATH)
            logging.info('exited get_best_model_from_gcloud in ModelEvaluation class')
            return result_model, best_model_path
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
        
    
    def mean_pool(self,token_embeds, attention_mask):
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool
    
    def encode_sentences(self, sentences, model, tokenizer):
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean pooling strategy for SBERT
            # embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = self.mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
        return embeddings
    
    def evaluate(self, model_path):
        try:
            logging.info('entered evaluate in ModelEvaluation class')
            
            model = transformers.BertModel.from_pretrained(model_path)
            tokenizer = transformers.BertTokenizer.from_pretrained(self.model_trainer_config.BASE_MODEL)
            
            validation_dataset = datasets.load_dataset('csv', data_files=self.model_trainer_artifacts.validation_dataset, split='train')
            validation_dataset = validation_dataset.select(range(100))
            validation_dataset = validation_dataset.map(lambda x: {'label': x['label'] / 5.0})
            
            sentence1_embeddings = self.encode_sentences(validation_dataset['sentence1'], model, tokenizer)
            sentence2_embeddings = self.encode_sentences(validation_dataset['sentence2'], model, tokenizer)
            cosine_similarities = util.pytorch_cos_sim(sentence1_embeddings, sentence2_embeddings)
            predicted_scores = cosine_similarities.diag().cpu().numpy()
            
            threshold = 0.85  # You may need to adjust this threshold based on your use case
            predicted_labels = (predicted_scores >= threshold).astype(int)

            # Convert ground truth similarity scores to binary labels
            ground_truth_scores = np.array(validation_dataset['label'])
            ground_truth_labels = (ground_truth_scores >= 3).astype(int)  # Using 3 as the threshold for similarity

            # Compute accuracy
            accuracy = accuracy_score(ground_truth_labels, predicted_labels)
            
            ground_truth_scores = validation_dataset['label']
            spearman_corr = spearmanr(predicted_scores, ground_truth_scores)
            
            print(f'Spearman correlation: {spearman_corr.correlation:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            logging.info('entered evaluate in ModelEvaluation class')
            return spearman_corr.correlation, accuracy
        except Exception as e:
            raise ExceptionHandler(e, sys) from e    

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info('entered initiate_model_evaluation in ModelEvaluation class')
            
            trained_model_spearman_correlation, trained_model_accuracy = self.evaluate(model_path = self.model_trainer_artifacts.trained_model_path)
            
            result_model, best_model_path = self.get_best_model_from_gcloud()
            if not result_model:
                is_model_accepted = True
                logging.info('there is no model in gcloud')
                print('there is no model in cloud')
            else:
                logging.info('model was downloaded from gcloud')
                logging.info('gcloud model is being evaluated')
                
                best_model_gcloud_spearman, best_model_gcloud_accuracy = self.evaluate(model_path=best_model_path)
                print('model was downloaded from cloud')
                print('model from cloud was evaluated')
                if best_model_gcloud_accuracy > trained_model_accuracy:
                    is_model_accepted = False
                    logging.info('Trained model is worse than gcloud one')
                    print('trained model is worse that cloud one')
                else:
                    is_model_accepted = True
                    logging.info('Trained model is better than gcloud one')
                    print('trained model is better than cloud one')
                
            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
                
            logging.info('exited initiate_model_evaluation in ModelEvaluation class')
            return model_evaluation_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e