from text_similarity.pipeline.train_pipeline import TrainingPipeLine
from text_similarity.pipeline.prediction_pipeline import PredictionPipeline

# obj = TrainingPipeLine()
# obj.run_pipeline()
pred = PredictionPipeline()
pred.run_pipeline('The diligent student pored over the ancient texts in the library.','The hardworking scholar studied the ancient manuscripts diligently.')