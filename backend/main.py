from fastapi import FastAPI, HTTPException
import torch
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from model import PointerNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Coordinates(BaseModel):
    points: List[List[float]]
    num_vehicles: int

# Cargar el modelo
def load_model():
    model = PointerNet(d=128)
    checkpoint = torch.load('vrp_model.pth', map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    else:
        state_dict = checkpoint
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

@app.post("/optimize")
async def optimize_route(coords: Coordinates):
    try:
        # Validaciones generales
        if len(coords.points) < 2:
            raise ValueError("Debe haber al menos un depósito y una ciudad.")
        if coords.num_vehicles not in [1, 2, 3]:
            raise ValueError("Número de vehículos debe ser entre 1 y 3.")

        points = np.array(coords.points, dtype=np.float32)
        depot = points[0]
        cities = points[1:]

        # === CASO 1 VEHÍCULO ===
        if coords.num_vehicles == 1:
            full_route = np.vstack([depot, cities])
            tensor = torch.tensor(full_route, dtype=torch.float32).unsqueeze(0)
            tour = model.greedy(tensor)
            distance = model.vrp_len(tensor, tour)

            t = tour.squeeze()
            if isinstance(t, torch.Tensor):
                indices = t.tolist() if t.ndim > 0 else [t.item()]
            elif isinstance(t, int):
                indices = [t]
            else:
                raise ValueError("Tipo inesperado de tour (ni tensor ni int)")

            indices = [i for i in indices if i > 0]

            if sorted(indices) != list(range(1, len(full_route))):
                raise ValueError("Ruta incompleta, el modelo omitió ciudades.")

            return {
                "routes": [{
                    "vehicle_id": 0,
                    "route": indices,
                    "distance": distance.item(),
                    "cities_visited": len(indices)
                }],
                "total_distance": distance.item()
            }

        # === CASO MULTI-VEHÍCULO ===
        kmeans = KMeans(n_clusters=coords.num_vehicles)
        clusters = kmeans.fit_predict(cities)
        routes = []
        total_distance = 0.0

        for i in range(coords.num_vehicles):
            cluster_points = cities[clusters == i]
            if len(cluster_points) == 0:
                continue

            vehicle_points = np.vstack([depot, cluster_points])
            tensor = torch.tensor(vehicle_points, dtype=torch.float32).unsqueeze(0)
            tour = model.greedy(tensor)
            distance = model.vrp_len(tensor, tour)

            t = tour.squeeze()
            if isinstance(t, torch.Tensor):
                idx_list = t.tolist() if t.ndim > 0 else [t.item()]
            elif isinstance(t, int):
                idx_list = [t]
            else:
                raise ValueError("Tipo inesperado de tour")

            indices = []
            for idx in idx_list:
                pt = vehicle_points[idx]
                dists = np.linalg.norm(cities - pt, axis=1)
                match = int(np.argmin(dists)) + 1
                indices.append(match)

            routes.append({
                "vehicle_id": i,
                "route": sorted(list(set(indices))),
                "distance": distance.item()
            })
            total_distance += distance.item()

        return {
            "routes": routes,
            "total_distance": total_distance
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}
