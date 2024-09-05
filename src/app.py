from reconocimientoFacial import captura_codificacion_rostros_video, entrenamiento_knn, face_rec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import shutil

app = FastAPI()

script_dir = os.path.dirname(os.path.abspath(__file__))

class VideoData(BaseModel):
    persona_id: str
    video_url: str

class KNNTrainingData(BaseModel):
    data_path: str
    model_save_path: str = None
    n_vecinos: int = None

class ReconocimientoData(BaseModel):
    image_source: str
    id_predict: str
    source_type: str  # Puede ser "url" o "base64"

@app.get("/get_all_training_status/")
async def get_all_training_status():
    path = os.path.join(script_dir, "data", "train")
    results = []
    try:
        for dir_name in os.listdir(path):
            dir_path = os.path.join(path, dir_name)
            if os.path.isdir(dir_path):
                files = os.listdir(dir_path)
                is_trained = len(files) > 0
                results.append({"id": dir_name, "is_trained": is_trained})

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_training_directory/{id}")
async def delete_training_directory(id: str):
    path = os.path.join(script_dir, "data", "train", id)
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No se encontró el directorio con id '{id}'")
    
    try:
        shutil.rmtree(path)
        return {"detail": f"Directorio '{id}' eliminado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/capturar_rostros_video/")
async def capturar_rostros_video(data: VideoData):
    success = captura_codificacion_rostros_video(data.persona_id, data.video_url)
    if not success:
        raise HTTPException(status_code=500, detail="Error capturando rostros desde el video.")
    return {"success": success}

@app.post("/entrenar_knn/")
async def entrenar_knn(data: KNNTrainingData):
    success = entrenamiento_knn(data.data_path, data.model_save_path, data.n_vecinos)
    if not success:
        raise HTTPException(status_code=500, detail="Error en el entrenamiento del modelo KNN.")
    return {"success": success}

@app.post("/reconocer_rostro/")
async def reconocer_rostro(data: ReconocimientoData):
    try:
        success = face_rec(data.image_source, data.id_predict, data.source_type)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# despliegue
# fastapi dev app.py --host 0.0.0.0 --port 8000
