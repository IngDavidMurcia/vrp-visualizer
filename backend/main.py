from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans

# Definición de la aplicación FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clase para la definición del modelo de entrada
class Coordinates(BaseModel):
    points: List[List[float]]
    num_vehicles: int

# Definición de la arquitectura del modelo (debe coincidir exactamente con la usada durante el entrenamiento)
class PointerNet(nn.Module):
    def __init__(self, d=128, nhead=8, num_layers=3):
        super(PointerNet, self).__init__()
        self.d = d
        
        # Capa de embedding inicial
        self.embedding = nn.Linear(2, d)
        
        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Capas para el decoder
        self.decoder = nn.LSTM(d, d, num_layers=1)
        self.pointer = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, 2)
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, d)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, d)
        
        # Encoder
        encoded = self.encoder(embedded)
        
        # Decoder (usamos el depósito como input inicial)
        decoder_input = encoded[0:1]  # Tomamos el depósito (primer elemento)
        hidden = None
        outputs = []
        
        for _ in range(seq_len - 1):  # -1 porque no necesitamos volver al depósito
            # LSTM step
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Attention
            expanded_output = output.expand(encoded.size(0), -1, -1)
            pointer_input = torch.cat([encoded, expanded_output], dim=-1)
            logits = self.pointer(pointer_input).squeeze(-1)
            
            # Mask para evitar repetir nodos
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for out in outputs:
                mask[out.argmax()] = True
            logits[mask] = -float('inf')
            
            # Selección
            outputs.append(logits)
            decoder_input = encoded[logits.argmax(dim=0)]
        
        return torch.stack(outputs)
    
    def greedy(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=0).squeeze()
    
    def vrp_len(self, points, tour):
        # points: (batch_size, seq_len, 2)
        # tour: (batch_size, seq_len)
        batch_size, seq_len = tour.shape
        
        # Calcular distancia entre puntos consecutivos
        total_distance = 0
        for i in range(batch_size):
            for j in range(seq_len - 1):
                p1 = points[i, tour[i, j]]
                p2 = points[i, tour[i, j+1]]
                total_distance += torch.norm(p1 - p2, p=2)
        
        return total_distance

# Carga del modelo
def load_model():
    model = PointerNet(d=128)
    try:
        checkpoint = torch.load('vrp_model.pth', map_location='cpu')
        
        # Manejar diferentes formatos de checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Eliminar 'module.' si el modelo fue entrenado con DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        # Cargar solo los pesos que coincidan
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
        
        model.eval()
        print("Modelo cargado correctamente")
        return model
    except Exception as e:
        print(f"Error cargando el modelo: {str(e)}")
        raise

# Cargar el modelo al iniciar
try:
    model = load_model()
except Exception as e:
    print(f"No se pudo cargar el modelo: {str(e)}")
    raise

# Validación de entrada
def validate_input(points, num_vehicles):
    if len(points) < 2:
        raise ValueError("Se necesitan al menos 2 puntos (depósito + al menos 1 ciudad)")
    if num_vehicles < 1 or num_vehicles > 3:
        raise ValueError("Número de vehículos debe ser entre 1 y 3")
    return True

# Endpoint principal

# Para múltiples vehículos
@app.post("/optimize")
async def optimize_route(coords: Coordinates):
    try:
        # Validación básica
        if len(coords.points) < 2:
            raise ValueError("Se necesitan al menos 2 puntos (depósito + al menos 1 ciudad)")
        if coords.num_vehicles < 1 or coords.num_vehicles > 3:
            raise ValueError("Número de vehículos debe ser entre 1 y 3")

        points = np.array(coords.points, dtype=np.float32)
        depot = points[0]
        cities = points[1:]
        num_cities = len(cities)

        # Caso para 1 vehículo - asegurar visita a todas las ciudades
        if coords.num_vehicles == 1:
            # Construir ruta completa: depósito + ciudades + depósito
            full_route = np.vstack([depot, cities, depot])
            points_tensor = torch.tensor(full_route, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                try:
                    # Obtener tour completo
                    tour = model.greedy(points_tensor)
                    
                    # Calcular distancia
                    distance = model.vrp_len(points_tensor, tour.unsqueeze(0))
                    
                    # Mapear índices a ciudades originales (ignorando depósitos)
                    city_indices = []
                    for idx in tour.tolist():
                        if 0 < idx < len(full_route) - 1:  # Ignorar depósitos
                            city_indices.append(idx - 1)  # Ajustar índice
                    
                    # Verificar que se visitaron todas las ciudades
                    if len(set(city_indices)) != num_cities:
                        missing = set(range(num_cities)) - set(city_indices)
                        raise ValueError(f"No se visitaron todas las ciudades. Faltan índices: {missing}")
                    
                    return {
                        "routes": [{
                            "vehicle_id": 0,
                            "route": city_indices,
                            "distance": distance.item(),
                            "cities_visited": len(set(city_indices))
                        }],
                        "total_distance": distance.item(),
                        "message": "Ruta optimizada para 1 vehículo"
                    }
                    
                except Exception as e:
                    raise ValueError(f"Error en optimización: {str(e)}")
        
        kmeans = KMeans(n_clusters=coords.num_vehicles)
        clusters = kmeans.fit_predict(cities)
        
        routes = []
        total_distance = 0.0
        
        for i in range(coords.num_vehicles):
            cluster_points = cities[clusters == i]
            if len(cluster_points) == 0:
                continue
                
            # Crear ruta para este vehículo (depósito + ciudades + depósito)
            vehicle_points = np.vstack([depot, cluster_points, depot])
            vehicle_tensor = torch.tensor(vehicle_points, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                tour = model.greedy(vehicle_tensor)
                distance = model.vrp_len(vehicle_tensor, tour.unsqueeze(0))
                
                # Mapear índices a los originales
                original_indices = []
                for idx in tour.tolist():
                    if idx == 0:  # Depósito inicial
                        original_indices.append(0)
                    elif idx == len(vehicle_points) - 1:  # Depósito final
                        original_indices.append(0)
                    else:
                        # Buscar ciudad en el listado original
                        city = vehicle_points[idx]
                        original_idx = np.where((cities == city).all(axis=1))[0][0] + 1
                        original_indices.append(original_idx)
                
                routes.append({
                    "vehicle_id": i,
                    "route": original_indices[1:-1],  # Excluir depósitos
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



# Endpoint de verificación
@app.get("/")
def health_check():
    return {
        "status": "API funcionando",
        "model_loaded": model is not None
    }

# Ejecutar con: uvicorn main:app --reload