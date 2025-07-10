import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './App.css';

function App() {
  const [route, setRoute] = useState(null);
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [mapInitialized, setMapInitialized] = useState(false);

  // Coordenadas originales (deben coincidir con las enviadas al backend)
  const originalCoordinates = [
    [40.7128, -74.0060], // Nueva York
    [34.0522, -118.2437], // Los Ángeles
    [41.8781, -87.6298]  // Chicago
  ];

  // 1. Inicializar el mapa
  useEffect(() => {
    if (!mapContainer.current) return;

    // 1. Validar que la API key esté configurada
    const mapTilerKey = 'wNHJXUkCv4sp6GAPuvsq'; // ¡API Key de mapTiler

    // 2. Inicializar mapa con manejo de errores
    try {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: `https://api.maptiler.com/maps/streets/style.json?key=${mapTilerKey}`,
        center: [-98.5833, 39.8333],
        zoom: 3
      });

      map.current.on('load', () => {
        console.log('Mapa cargado correctamente');
        setMapInitialized(true);
      });

      map.current.on('error', (e) => {
        console.error('Error en el mapa:', e.error);
      });

    } catch (error) {
      console.error('Error al crear el mapa:', error);
    }

    return () => {
      if (map.current) map.current.remove();
    };
  }, []);

  // 2. Dibujar ruta cuando hay datos
  useEffect(() => {
    if (!mapInitialized || !route || !map.current) {
      console.log('Condiciones no cumplidas:', {
        initialized: mapInitialized,
        route: !!route,
        map: !!map.current
      });
      return;
    }

    try {
      // Verificar que el mapa esté realmente cargado
      if (!map.current.loaded()) {
        console.warn('Mapa no está listo');
        return;
      }

      // Convertir índices a coordenadas [lon, lat]
      const routeCoords = route.route.map(index => {
        const [lat, lon] = originalCoordinates[index];
        return [lon, lat];
      });

      // Limpiar capa anterior
      if (map.current.getSource('route')) {
        map.current.removeLayer('route');
        map.current.removeSource('route');
      }

      // Añadir nueva ruta
      map.current.addSource('route', {
        type: 'geojson',
        data: {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: routeCoords
          }
        }
      });

      map.current.addLayer({
        id: 'route',
        type: 'line',
        source: 'route',
        paint: {
          'line-color': '#FF0000',
          'line-width': 4
        }
      });

      // Añadir marcadores
      originalCoordinates.forEach(([lat, lon], i) => {
        new maplibregl.Marker()
          .setLngLat([lon, lat])
          .setPopup(new maplibregl.Popup().setText(`Punto ${i}`))
          .addTo(map.current);
      });

      // Ajustar vista
      const bounds = new maplibregl.LngLatBounds();
      routeCoords.forEach(coord => bounds.extend(coord));
      map.current.fitBounds(bounds, { padding: 50 });

    } catch (error) {
      console.error('Error al dibujar ruta:', error);
    }
  }, [route, mapInitialized]);

  const optimizeRoute = async () => {
    try {
      const response = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: originalCoordinates })
      });
      const data = await response.json();
      console.log("Datos recibidos:", data);
      setRoute(data);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="app-container">
      <h1>Optimizador de Rutas VRP</h1>
      <button onClick={optimizeRoute}>Calcular Ruta</button>
      <div 
        ref={mapContainer} 
        className="map-container" 
        style={{ 
          height: '500px', 
          width: '100%',
          border: '1px solid red' // Temporal para debug
        }}
      />
      {route && (
        <div className="route-info">
          <h3>Ruta: {route.route.join(" → ")}</h3>
          <p>Distancia: {(route.distance / 1000).toFixed(2)} km</p>
        </div>
      )}
    </div>
  );
}

export default App;