from fastapi import FastAPI, HTTPException
import torch
import numpy as np
from pydantic import BaseModel
from typing import List
from model import PointerNet
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans

# Cargar modelo entrenado
model = PointerNet(d=128)
model_data = torch.load('vrp_model.pth', map_location='cpu')
model.load_state_dict(model_data['model_state_dict'])
model.eval()

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class Coordinates(BaseModel):
    points: List[List[float]]
    num_vehicles: int = 1

def cluster_points(points, n_clusters):
    """Divide puntos en clusters usando K-Means"""
    points_array = np.array(points[1:])  # Excluye el depósito
    if len(points_array) < n_clusters:
        n_clusters = len(points_array)
    
    if n_clusters <= 1:
        return [points_array]
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(points_array)
    return [points_array[clusters == i] for i in range(n_clusters)]

@app.post("/optimize")
async def optimize_route(coords: Coordinates):
    try:
        points = coords.points
        num_vehicles = coords.num_vehicles
        
        # Redondear coordenadas para evitar problemas de precisión
        rounded_points = [[round(lat, 6), round(lon, 6)] for lat, lon in points]
        
        # Convertir a tensor
        points_tensor = torch.tensor(rounded_points, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            tour = model.greedy(points_tensor)[0]
            distance = model.vrp_len(points_tensor, tour.unsqueeze(0)).item()
            
            # Mapear índices a los puntos originales
            original_indices = []
            for idx in tour.tolist():
                if idx == 0:  # Depósito
                    original_indices.append(0)
                    continue
                    
                # Buscar el punto correspondiente con tolerancia
                point = points_tensor[0][idx].tolist()
                found = False
                for i, coord in enumerate(points):
                    if (abs(coord[0] - point[0]) < 0.0001 and 
                        abs(coord[1] - point[1]) < 0.0001):
                        original_indices.append(i)
                        found = True
                        break
                
                if not found:
                    original_indices.append(idx)  # Usar índice original si no se encuentra
            
        return {
            "route": original_indices,
            "distance": distance,
            "vehicle_routes": [original_indices]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "API lista con modelo cargado"}