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


  // Coordenadas originales (deben coincidir con las enviadas al backend)
  //forma: [latitud, longitud]
  const originalCoordinates = [
    [6.2318, -75.5636], // Medellín
    [4.639, -74.0817], // Bogotá
    [4.4353, -75.2110]// Ibagué
  ];
/*
[6.2318, -75.5636], // Medellín
[4.639, -74.0817], // Bogotá
[4.4353, -75.2110]// Ibagué
[10.9639, -74.7964] // Barranquilla
[3.4372, -76.5225], // Cali
[8.3639, -75.4000], // Manizales
[7.1233, -73.1216], // Bucaramanga
[10.9886, -74.8031], // Cartagena
[5.0692, -75.5144], // Pereira
*/

const clearRoute = () => {
  setRoute(null);
  if (map.current?.getSource('route')) {
    map.current.removeLayer('route');
    map.current.removeSource('route');
  }
};

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
        center: [-74.2973, 4.5709], // Centro de Colombia
        zoom: 5.5
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
      // Validar que hay puntos para enviar
      if (originalCoordinates.length === 0) {
        alert("Agrega puntos primero");
        return;
      }
  
      const response = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          points: originalCoordinates,
          num_vehicles: 1 // Puede hacerse dinámico después
        })
      });
  
      if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);
      
      const data = await response.json();
      console.log("Ruta optimizada:", data);
      setRoute(data);
      
    } catch (error) {
      console.error("Error al calcular ruta:", error);
      alert(`Error: ${error.message}`);
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
      {!mapInitialized && (
        <div style={{color: 'red', padding: '10px'}}>
        Mapa no inicializado. Revisa la consola para errores.
        </div>
                          )} 
    </div>
  );
}

export default App;