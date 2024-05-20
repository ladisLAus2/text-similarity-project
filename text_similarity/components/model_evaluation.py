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

    def get_best_model_from_gcloud(self) -> str:
        try:
            logging.info('entered get_best_model_from_gcloud in ModelEvaluation class')
            os.makedirs(self.model_evaluation_config.BEST_MODEL_PATH, exist_ok=True)
            self.gcloud.syncronize_from_cloud_to_folder(
                self.model_evaluation_config.BUCKET_NAME,
                self.model_evaluation_config.MODEL_NAME,
                self.model_evaluation_config.BEST_MODEL_PATH
            )
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_PATH, self.model_evaluation_config.MODEL_NAME)
            logging.info('exited get_best_model_from_gcloud in ModelEvaluation class')
            return best_model_path
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
    
    def encode_sentences(self, sentences, model, tokenizer):
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean pooling strategy for SBERT
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    
    def evaluate(self, model_path):
        try:
            logging.info('entered evaluate in ModelEvaluation class')
            
            model = transformers.BertModel.from_pretrained(model_path)
            tokenizer = transformers.BertTokenizer.from_pretrained(self.model_trainer_config.BASE_MODEL)
            
            validation_dataset = datasets.load_dataset('csv', data_files=self.model_trainer_artifacts.validation_dataset, split='train')
            validation_dataset = validation_dataset.map(lambda x: {'label': x['label'] / 5.0})
            
            sentence1_embeddings = self.encode_sentences(validation_dataset['sentence1'], model, tokenizer)
            sentence2_embeddings = self.encode_sentences(validation_dataset['sentence2'], model, tokenizer)
            cosine_similarities = util.pytorch_cos_sim(sentence1_embeddings, sentence2_embeddings)
            predicted_scores = cosine_similarities.diag().cpu().numpy()
            ground_truth_scores = validation_dataset['label']
            spearman_corr = spearmanr(predicted_scores, ground_truth_scores)
            threshold = 0.5
            predicted_labels = (predicted_scores >= threshold).astype(int)
            ground_truth_scores = np.array(validation_dataset['label'])
            ground_truth_labels = (ground_truth_scores >= 3).astype(int)
            accuracy = accuracy_score(ground_truth_labels, predicted_labels)
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
            
            best_model_path = self.get_best_model_from_gcloud()
            
            if not len(os.listdir(best_model_path)) == 0:
                is_model_accepted = True
                logging.info('there is no model in gcloud')
            else:
                logging.info('model was downloaded from gcloud')
                best_model_gcloud = transformers.BertModel.from_pretrained(best_model_path)
                logging.info('gcloud model is being evaluated')
                best_model_gcloud_spearman, best_model_gcloud_accuracy = self.evaluate(model_path=best_model_gcloud)
                
                if best_model_gcloud_spearman > trained_model_spearman_correlation:
                    is_model_accepted = True
                    logging.info('Trained model is worse than gcloud one')
                else:
                    is_model_accepted = False
                    logging.info('Trained model is better than gcloud one')
                
            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
                
            logging.info('exited initiate_model_evaluation in ModelEvaluation class')
            return model_evaluation_artifacts
        except Exception as e:
            raise ExceptionHandler(e, sys) from e