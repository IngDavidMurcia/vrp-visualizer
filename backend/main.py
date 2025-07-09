from fastapi import FastAPI, HTTPException  #FastAPI para crear la API y HTTPException para manejar errores
import torch # PyTorch para el modelo de optimización
import numpy as np # NumPy para operaciones numéricas
from pydantic import BaseModel # Pydantic para validación de datos
from typing import List # List para manejar listas de puntos

app = FastAPI() # Instancia de la aplicación FastAPI

class Coordinates(BaseModel): 
    points: List[List[float]]  # Formato: [[lat1, lon1], [lat2, lon2], ...]

# Mock del modelo (Se reemplazara con tu modelo real)
model = torch.nn.Module()  # <-- poner modelo PyTorch aquí

@app.post("/optimize")
async def optimize_route(coords: Coordinates):
    try:
        points = torch.tensor(coords.points, dtype=torch.float32)
        
        # Simulación de predicción (reemplazar esto, solo mientras esta listo el modelo)
        mock_route = [1, 0, 2] if len(coords.points) >= 3 else list(range(len(coords.points)))
        mock_distance = 123.45
        
        return {
            "route": mock_route,
            "distance": mock_distance,
            "vehicle_routes": [mock_route]  # Para múltiples vehículos
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "API lista para Checkpoint 3"}