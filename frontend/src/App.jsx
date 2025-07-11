import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './App.css';

function App() {
  const [route, setRoute] = useState(null);
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [mapInitialized, setMapInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [numVehicles, setNumVehicles] = useState(1);

  // Coordenadas de Colombia
  // (el primer punto es el depósito/centro de acopio)
const originalCoordinates = [
  [4.5709, -74.2973],  // Depósito (Bogotá centro)
  [6.2318, -75.5636],  // Medellín
  [4.639, -74.0817],   // Bogotá (otro punto)
  [4.4353, -75.2110]   // Ibagué
];

  const clearRoute = () => {
    setRoute(null);
    try {
    if (map.current?.getSource('route')) {
      map.current.removeLayer('route');
      map.current.removeSource('route');
    }
  } catch (error) {
    console.log("Error al limpiar ruta (puede ignorarse):", error.message);
  }
  };

  // 1. Inicializar el mapa
  useEffect(() => {
    if (!mapContainer.current) return;

    const mapTilerKey = 'wNHJXUkCv4sp6GAPuvsq'; // API Key

    try {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: `https://api.maptiler.com/maps/streets/style.json?key=${mapTilerKey}`,
        center: [-74.2973, 4.5709], // Centro de Colombia
        zoom: 4
      });

      map.current.on('load', () => {
        console.log('Mapa cargado correctamente');
        setMapInitialized(true);
        
        // Añadir marcadores iniciales
        originalCoordinates.forEach(([lat, lon], i) => {
        // Crear elemento HTML personalizado para el marcador
        const markerElement = document.createElement('div');
        markerElement.className = 'custom-marker';
  
        // Texto diferente para depósito vs puntos
        const markerText = i === 0 ? 'A' : `P${i}`; // "A" para acopio, "P1", "P2", etc.
  
        markerElement.innerHTML = `
          <div class="marker-container">
          <div class="marker-circle" style="background: ${i === 0 ? '#FFA500' : '#3FB1CE'}">
          ${markerText}
          </div>
          </div>
          `;

          new maplibregl.Marker({element: markerElement})
            .setLngLat([lon, lat])
            .setPopup(new maplibregl.Popup().setText(i === 0 ? 'Centro de acopio' : `Punto ${i}`))
            .addTo(map.current);
        });
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
      // 1. Obtener coordenadas completas [lon, lat]
      const getCoords = (index) => {
        const [lat, lon] = originalCoordinates[index];
        return [lon, lat];
      };

      // 2. Añadir el depósito al inicio y final de la ruta
      const routeCoords = route.route.map(index => getCoords(index));
      const fullRoute = [
        getCoords(0), // Depósito inicial
        ...routeCoords,
        getCoords(0)  // Depósito final
      ];

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
            coordinates: fullRoute // Usar fullRoute para incluir el depósito, sino usar routeCoords
          }
        }
      });
      // Añadir capa de ruta
      map.current.addLayer({
        id: 'route',
        type: 'line',
        source: 'route',
        paint: {
          'line-color': '#FF0000',
          'line-width': 4,
          'line-dasharray': [2, 2] // Opcional: estilo de línea punteada
        }
      });
      // Añadir marcadores (incluyendo depósito con estilo diferente)
      originalCoordinates.forEach(([lat, lon], i) => {
        const marker = new maplibregl.Marker({
          color: i === 0 ? '#FFA500' : '#3FB1CE'  // Naranja para depósito
      })  
          .setLngLat([lon, lat])
          .setPopup(new maplibregl.Popup().setText(
            i === 0 ? 'Centro de acopio' : `Punto ${i}`
        ))
          .addTo(map.current);
      });

      /*
      map.current.addLayer({
        id: 'route',
        type: 'line',
        source: 'route',
        paint: {
          'line-color': '#FF0000',
          'line-width': 4
        }
      });*/


      // Ajustar vista
      const bounds = new maplibregl.LngLatBounds();
      fullRoute.forEach(coord => bounds.extend(coord));
      map.current.fitBounds(bounds, { padding: 50 });

    } catch (error) {
      console.error('Error al dibujar ruta:', error);
    }
  }, [route, mapInitialized]);

  const optimizeRoute = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          points: originalCoordinates,
          num_vehicles: numVehicles
        })
      });

      if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);
      
      const data = await response.json();
      setRoute(data);
      
    } catch (error) {
      console.error("Error:", error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>Optimizador IA de Rutas (VRP)</h1>
      
      <div className="controls">
        <select 
          value={numVehicles} 
          onChange={(e) => setNumVehicles(parseInt(e.target.value))}
        >
          <option value={1}>1 Vehículo</option>
          <option value={2}>2 Vehículos</option>
          <option value={3}>3 Vehículos</option>
        </select>

        <button onClick={optimizeRoute} disabled={isLoading}>
          {isLoading ? 'Calculando...' : 'Calcular Ruta'}
        </button>
        
        <button onClick={clearRoute} className="secondary-btn">
          Limpiar Ruta
        </button>
      </div>

      <div 
        ref={mapContainer} 
        className="map-container" 
        style={{ border: '2px solid #ccc' }}
      />

      {route && (
        <div className="route-info">
          <h3>Ruta optimizada:</h3>
          <p><strong>Secuencia:</strong> Depósito → {route.route.map(i => `Punto ${i}`).join(" → ")} → Depósito</p>
          <p><strong>Distancia total:</strong> {(route.distance / 1000).toFixed(2)} km</p>
        
          <div className="route-details">
            <h4>Coordenadas:</h4>
              <ul>
                <li><strong>Depósito:</strong> {originalCoordinates[0][0]}, {originalCoordinates[0][1]}</li>
                {route.route.map((index, i) => (
                <li key={i}>
                <strong>Punto {index}:</strong> {originalCoordinates[index][0]}, {originalCoordinates[index][1]}
                </li>
                ))}
              </ul>
          </div>
        </div>
      )}

      {!mapInitialized && (
        <div className="map-loading">
          Cargando mapa... (Si no aparece, revisa la consola)
        </div>
      )}
    </div>
  );
}

export default App;