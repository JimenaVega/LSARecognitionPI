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


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(video: UploadFile = File(...)):
    # Crea un archivo temporal para guardar el video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
        # Guarda el contenido del archivo subido en el archivo temporal
        shutil.copyfileobj(video.file, tmp_file)
        tmp_file_path = tmp_file.name

    print('Starting video process...')
    
    sign = process_sign(tmp_file_path)

    print('Video process completed...')

    # # Abre el video con OpenCV
    # cap = cv2.VideoCapture(tmp_file_path)
    # if not cap.isOpened():
    #     raise ValueError("Error al abrir el archivo de video con OpenCV.")
    
    # while cap.isOpened():
    #     ret, frame = cap.read()

    #     if ret:
    #         cv2.imshow('Webcam Feed', frame)

    #     # Wait for a key to be pressed.
    #     if cv2.waitKey(10) & 0xFF == ord("q"):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # Elimina el archivo temporal
    os.remove(tmp_file_path)

    sign = sign_map[str(sign)]

    return {"message": f"Sign recognized: {sign}"}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
