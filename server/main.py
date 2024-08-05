import uvicorn
import json
import tempfile
import os
import sys
import shutil

from typing import Union
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware

sys.path.append('/home/alejo/repos/LSARecognitionPI/server')

from utils import process_sign

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Replace "*" with specific methods if needed
    allow_headers=["*"],  # Replace "*" with specific headers if needed
)

with open('labels.json') as json_file:
    sign_map = json.load(json_file)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/status/")
def get_status():
    return {"status": 1}

@app.post("/predict/")
def predict_sign(video: UploadFile = File(...)):
    # Crea un archivo temporal para guardar el video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
        # Guarda el contenido del archivo subido en el archivo temporal
        shutil.copyfileobj(video.file, tmp_file)
        tmp_file_path = tmp_file.name

    print('Starting video process...')
    
    sign = process_sign(tmp_file_path)

    print('Video process completed...')

    os.remove(tmp_file_path)

    sign = sign_map[str(sign)]

    return {"sign": sign}


if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="/home/alejo/repos/LSARecognitionPI/server/certs/server.key",
        ssl_certfile="/home/alejo/repos/LSARecognitionPI/server/certs/server.crt"
    )
