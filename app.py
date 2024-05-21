from text_similarity.pipeline.train_pipeline import TrainingPipeLine
from text_similarity.pipeline.prediction_pipeline import PredictionPipeline

from fastapi import FastAPI
import uvicorn
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from text_similarity.exception import ExceptionHandler
from text_similarity.constants import *

sen1: str= 'The diligent student pored over the ancient texts in the library.'
sen2: str= 'The hardworking scholar studied the ancient manuscripts diligently.'


app = FastAPI()

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')

@app.get('/train')
async def training():
    try:
        training_pipe = TrainingPipeLine()
        
        training_pipe.run_pipeline()

        return Response('Training was successful')
        
    except Exception as e:
        return Response(f'Error happened! {e}')
    
@app.post('/predict')
async def predicting(sentence_1, sentence_2):
    try:
        print(sentence_1, type(sentence_2))
        predicting_pipe = PredictionPipeline()
        similarity = predicting_pipe.run_pipeline(sentence_1, sentence_2)
        
        return similarity.item()
    except Exception as e:
        raise ExceptionHandler(e, sys) from e
    
if __name__=='__main__':
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
    
    