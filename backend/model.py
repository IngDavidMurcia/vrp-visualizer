import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import numpy as np

# Parámetros del problema (deben coincidir con tu notebook)
DEPOT_IDX = 0
K = 3  # Número de clientes por vehículo
FACTOR_KM = 75

class Encoder(nn.Module):
    def __init__(self, d=128, h=8, l=3):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d, nhead=h, batch_first=True) 
            for _ in range(l)
        ])
    
    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h)
        return h

class Decoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.scale = d ** -0.5
    
    def forward(self, enc, node, mask):
        q = self.q(enc[torch.arange(enc.size(0),device=enc.device), node]).unsqueeze(1)
        k = self.k(enc)
        logits = (q @ k.transpose(-2, -1)).squeeze(1) * self.scale
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

class PointerNet(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.enc = Encoder(d)
        self.dec = Decoder(d)
    
    @torch.no_grad()
    def greedy(self, c):
        """Genera rutas usando decodificación greedy como en el notebook)"""
        b, n, _ = c.size()
        e = self.enc(c)
        vis = torch.zeros(b, n, dtype=torch.bool, device=c.device)
        vis[:, DEPOT_IDX] = True  # Marcar depósito como visitado
        
        tour = torch.empty(b, n-1, dtype=torch.long, device=c.device)  # n-1 clientes
        node = torch.randint(1, n, (b,), device=c.device)  # Inicio aleatorio
        
        for s in range(n-1):  # Para n-1 clientes (excluyendo depósito)
            tour[:, s] = node
            mask = vis.clone()
            mask[torch.arange(b), node] = True  # Marcar nodo actual
            
            # Seleccionar próximo nodo con mayor atención
            node = self.dec(e, node, mask).argmax(-1)
            vis = mask
        
        return tour
    
    def vrp_len(self, c, tour):
        """Calcula distancia total (como en la funcion vrp_len del modelo)"""
        b = c.size(0)
        d = c[:, DEPOT_IDX:DEPOT_IDX+1]  # Coordenadas del depósito
        tot = torch.zeros(b, device=c.device)
        
        for s in range(0, tour.size(1), K):  # K=3 clientes por vehículo
            seg = tour[:, s:s+K]
            # Ruta: depósito -> clientes -> depósito
            a = torch.cat([
                d.expand(-1, 1, -1),
                c[torch.arange(b).unsqueeze(1), seg],
                d.expand(-1, 1, -1)
            ], dim=1)
            tot += torch.norm(a[:, :-1] - a[:, 1:], dim=-1).sum(-1)
        
        return tot * FACTOR_KM  # Convertir a km