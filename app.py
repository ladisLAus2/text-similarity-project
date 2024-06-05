from text_similarity.pipeline.train_pipeline import TrainingPipeLine
from text_similarity.pipeline.prediction_pipeline import PredictionPipeline
from fastapi import FastAPI, File, UploadFile, HTTPException,WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from text_similarity.exception import ExceptionHandler
from text_similarity.constants import *
from fastapi.middleware.cors import CORSMiddleware
from text_similarity.configuration.gcloud_syncer import GCloudSyncher
import tempfile
app = FastAPI()

origins = [
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        
        return {"similarity": similarity.item()}
    except Exception as e:
        raise ExceptionHandler(e, sys) from e
    
    
class UploadResponse(BaseModel):
    success: bool
    message: str
    
    
syncher = GCloudSyncher()

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed.")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    upload_success = syncher.upload_to_cloud(file.filename, temp_file_path, BUCKET_NAME)
    
    if upload_success:
        os.remove(temp_file_path)
        return UploadResponse(success=True, message="File uploaded successfully")
    else:
        os.remove(temp_file_path)
        return UploadResponse(success=False, message="Failed to upload file")


@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        training_pipeline = TrainingPipeLine()

        for message in training_pipeline.run_pipeline_with_progress():
            print(message)
            await websocket.send_text(message)

        await websocket.send_text("Training was successful")
    except Exception as e:
        await websocket.send_text(f"Error happened! {e}")
    finally:
        await websocket.close()


    
if __name__=='__main__':
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
    
    