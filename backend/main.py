from fastapi import FastAPI, HTTPException  #FastAPI para crear la API y HTTPException para manejar errores
import torch # PyTorch para el modelo de optimización
import numpy as np # NumPy para operaciones numéricas
from pydantic import BaseModel # Pydantic para validación de datos
from typing import List # List para manejar listas de puntos
from model import PointerNet # Importar el modelo PointerNet desde model.py

# --- Bloque de diagnóstico ---  Porque no me funciono lacarga del .pth
print("\n Verificando estructura del archivo .pth:")
model_data = torch.load('vrp_model.pth', map_location='cpu')
print("Claves en el archivo:", model_data.keys())  # Deberías ver ['model_state_dict', 'config']
print("Tipo de 'model_state_dict':", type(model_data['model_state_dict']))
print("Ejemplo de pesos cargados (5 primeros):", list(model_data['model_state_dict'].keys())[:5])
# ----------------------------



# Cargar modelo entrenado
model = PointerNet(d=128) #mismas dimensiones que en el notebook 
model_data = torch.load('vrp_model.pth', map_location='cpu')
model.load_state_dict(model_data['model_state_dict'])  # Accede al state_dict real
model.eval()
# ------------------------------------

app = FastAPI() # Instancia de la aplicación FastAPI

class Coordinates(BaseModel): 
    points: List[List[float]]  # Formato: [[lat1, lon1], [lat2, lon2], ...]

# Ruta para optimizar la ruta de vehículos
@app.post("/optimize")
async def optimize_route(coords: Coordinates):
    try:
        points = torch.tensor(coords.points, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
        
        with torch.no_grad():
            tour = model.greedy(points)[0]  # [0] para remover batch dim
            distance = model.vrp_len(points, tour.unsqueeze(0)).item()
        
        return {
            "route": tour.tolist(),
            "distance": distance,
            "vehicle_routes": [tour.tolist()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "API lista con modelo cargado"}
# ------------------------------------