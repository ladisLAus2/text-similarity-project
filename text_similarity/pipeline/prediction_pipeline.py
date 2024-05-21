import os
import io
import sys
from PIL import Image
from text_similarity.logger import logging
from text_similarity.exception import ExceptionHandler
from text_similarity.configuration.gcloud_syncer import GCloudSyncher
from text_similarity.components.data_transformation import DataTransformation
from text_similarity.entity.config_entity import DataTransformationConfig
from text_similarity.entity.artifact_entity import DataIngestionArtifacts
from text_similarity.constants import *
import transformers
import torch
from sentence_transformers import util

class PredictionPipeline:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.MODEL_NAME = TRAINED_MODEL_NAME
        self.CONFIG = TRAINED_MODEL_CONFIG
        self.BASE_MODEL = BASE_MODEL
        self.PREDICTION_MODEL_PATH = os.path.join('artifacts','prediction_model')
        self.CLOUD = GCloudSyncher()
    
    
    def get_model_from_cloud(self):
        try:
            logging.info('entered get_model_from_cloud in PredictionPipeline class')
            os.makedirs(self.PREDICTION_MODEL_PATH, exist_ok=True)
            if os.path.exists(os.path.join(self.PREDICTION_MODEL_PATH, self.MODEL_NAME)):
                local = self.CLOUD.generate_md5(os.path.join(self.PREDICTION_MODEL_PATH, self.MODEL_NAME))
                print(os.path.join(self.PREDICTION_MODEL_PATH, self.MODEL_NAME))
                gcloud = self.CLOUD.get_gcs_file_md5(self.BUCKET_NAME, self.MODEL_NAME)
                print(local, gcloud)
                if not (local == gcloud):
                    result_model = self.CLOUD.download_file_from_cloud(self.MODEL_NAME,os.path.join(self.PREDICTION_MODEL_PATH, self.MODEL_NAME), self.BUCKET_NAME)
                    result_config = self.CLOUD.download_file_from_cloud(self.CONFIG,os.path.join(self.PREDICTION_MODEL_PATH, self.CONFIG), self.BUCKET_NAME)
                    
                    if result_model and result_config:
                        print('model and config were downloaded from cloud')
                else: 
                    print('the local model is the same as cloud one')
            logging.info('exited from get_model_from_cloud in PredictionPipeline class')
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

    def predict(self, sentence_1, sentence_2):
        try:
            logging.info('entered predict method in PredictionPipeline class')
            
            self.get_model_from_cloud()
            model = transformers.BertModel.from_pretrained(self.PREDICTION_MODEL_PATH)
            tokenizer = transformers.BertTokenizer.from_pretrained(self.BASE_MODEL)
            embeddings_1 = self.encode_sentences(sentence_1,model, tokenizer)
            embeddings_2 = self.encode_sentences(sentence_2, model, tokenizer)
            cosine_similarities = util.pytorch_cos_sim(embeddings_1, embeddings_2)
            return cosine_similarities
            logging.info('exited predict method in PredictionPipeline class')
        except Exception as e:
            raise ExceptionHandler(e, sys) from e
        
    def run_pipeline(self, sentence_1, sentence_2):
        try:
            logging.info('entererd run_pipeline in PredictionPipeline class')
            
            similarity = self.predict(sentence_1, sentence_2)
            print(similarity)
            logging.info('entererd run_pipeline in PredictionPipeline class')
            return similarity
        except Exception as e:
            raise ExceptionHandler(e, sys) from e