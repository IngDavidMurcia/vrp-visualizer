import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './App.css';
import './mapPopupOverrides.css';
import gsap from 'gsap';
import * as turf from '@turf/turf';

// Listado predefinido de ciudades (7 Colombia + 3 EEUU)
const citiesList = [
  { name: "Bogotá (Centro)", coords: [4.5709, -74.2973], isDepot: true },//Centro de Acopio y capital
  { name: "Medellín", coords: [6.2318, -75.5636] },//Capital del Eje Cafetero, ciudad principal
  { name: "Cali", coords: [3.4516, -76.5320] },// Capital del Valle del Cauca, ciudad principal
  { name: "Barranquilla", coords: [10.9639, -74.7964] },//Caribe, ciudad principal
  { name: "Cartagena", coords: [10.3910, -75.4794] },// Caribe, ciudad principal
  { name: "Bucaramanga", coords: [7.1193, -73.1227] },//  Santander, ciudad principal
  { name: "Pereira", coords: [4.8143, -75.6946] },//  Eje Cafetero, ciudad principal
  { name: "Santa Marta", coords: [11.2408, -74.1990] }, // Caribe
  { name: "Manizales", coords: [5.0689, -75.5174] }, // Eje Cafetero
  { name: "Cúcuta", coords: [7.8939, -72.5078] }, // Frontera con Venezuela
  { name: "Villavicencio", coords: [4.1420, -73.6266] }, // Llanos Orientales
  { name: "Ibagué", coords: [4.4375, -75.2006] }, // Tolima
  { name: "Armenia", coords: [4.5380, -75.6721] } // Quindío
];

function App() {
  const map = useRef(null);
  const mapContainer = useRef(null);
  const [selectedCities, setSelectedCities] = useState([citiesList[0]]);
  const [numAgents, setNumAgents] = useState(1);
  const [route, setRoute] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mapInitialized, setMapInitialized] = useState(false);
  const markersRef = useRef([]);
  const animationRefs = useRef({});
  const vehicleMarkersRef = useRef({});
  const animationDataRef = useRef({});
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);

  useEffect(() => {
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [-75.56359, 6.25184],
      zoom: 12
    });

    map.current.on('load', () => {
      setMapInitialized(true);
    });

    return () => map.current.remove();
  }, []);

  useEffect(() => {
    if (mapInitialized) updateMapMarkers();
  }, [selectedCities, mapInitialized]);

  const updateMapMarkers = () => {
    if (!map.current) return;
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    selectedCities.forEach((city, i) => {
      const el = document.createElement('div');
      el.className = 'custom-marker';
      el.innerHTML = `
        <div class="marker-container">
          <div class="marker-pin" style="background: \${city.isDepot ? '#FFA500' : '#3FB1CE'}"></div>
          <div class="marker-label">\${city.isDepot ? 'A' : 'P' + i}</div>
        </div>`;

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([city.coords[1], city.coords[0]])
        .addTo(map.current);

      markersRef.current.push(marker);
    });
  };

  const removeRouteLayers = () => {
    for (let i = 0; i < 3; i++) {
      if (map.current.getLayer('route-line-' + i)) map.current.removeLayer('route-line-' + i);
      if (map.current.getSource('route-' + i)) map.current.removeSource('route-' + i);
    }
  };

  const clearRoute = () => {
    setRoute(null);
    removeRouteLayers();
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];
    Object.values(vehicleMarkersRef.current).forEach(m => m.remove());
    vehicleMarkersRef.current = {};
    animationRefs.current = {};
    animationDataRef.current = {};
    setIsAnimating(false);
  };

  const optimizeRoute = async () => {
    if (selectedCities.length < 2) return alert('Selecciona al menos un centro de acopio y una ciudad.');
    setIsLoading(true);
    try {
      const res = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          points: selectedCities.map(c => [c.coords[0], c.coords[1]]),
          num_vehicles: numAgents
        })
      });
      const data = await res.json();
      setRoute(data);
      drawRoutes(data);
    } catch (e) {
      console.error('Error:', e);
    } finally {
      setIsLoading(false);
    }
  };

  const drawRoutes = (data) => {
    if (!map.current) return;
    removeRouteLayers();

    const features = data.routes.map((r, i) => {
      const coords = r.route.map(idx => [selectedCities[idx].coords[1], selectedCities[idx].coords[0]]);
      coords.unshift([selectedCities[0].coords[1], selectedCities[0].coords[0]]);
      coords.push([selectedCities[0].coords[1], selectedCities[0].coords[0]]);

      map.current.addSource('route-' + i, {
        type: 'geojson',
        data: {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: coords
          }
        }
      });

      map.current.addLayer({
        id: 'route-line-' + i,
        type: 'line',
        source: 'route-' + i,
        paint: {
          'line-color': ['red', 'blue', 'green'][i % 3],
          'line-width': 4
        }
      });

      return {
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: coords
        }
      };
    });

    prepareAnimation(features);
  };

  const prepareAnimation = (routesGeoJson) => {
    animationRefs.current = {};
    vehicleMarkersRef.current = {};
    animationDataRef.current = {};

    routesGeoJson.forEach((routeFeature, vehicleIdx) => {
      const route = routeFeature.geometry;
      const line = turf.lineString(route.coordinates);
      const totalLength = turf.length(line, { units: 'kilometers' });

      const steps = Math.floor(totalLength / 0.05);
      const points = [];
      for (let i = 0; i <= steps; i++) {
        const segment = turf.along(line, (totalLength / steps) * i, { units: 'kilometers' });
        points.push(segment.geometry.coordinates);
      }

      const el = document.createElement('div');
      el.className = 'animated-vehicle';
      el.style.width = '14px';
      el.style.height = '14px';
      el.style.borderRadius = '50%';
      el.style.backgroundColor = ['red', 'blue', 'green'][vehicleIdx % 3];
      el.style.boxShadow = '0 0 12px rgba(0,0,0,0.5)';

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat(points[0])
        .addTo(map.current);

      vehicleMarkersRef.current[vehicleIdx] = marker;
      animationDataRef.current[vehicleIdx] = { points, idx: 0 };
    });
  };

  const startAnimation = () => {
    if (!map.current || isAnimating) return;
    setIsAnimating(true);

    Object.entries(animationDataRef.current).forEach(([vehicleIdx, data]) => {
      const marker = vehicleMarkersRef.current[vehicleIdx];
      const { points } = data;
      const duration = (points.length / animationSpeed) * 0.008;

      animationRefs.current[vehicleIdx] = gsap.to({ index: 0 }, {
        index: points.length - 1,
        duration,
        ease: 'none',
        onUpdate: function () {
          const i = Math.round(this.targets()[0].index);
          if (points[i]) marker.setLngLat(points[i]);
        },
        onComplete: () => setIsAnimating(false)
      });
    });
  };

  const pauseAnimation = () => {
    Object.values(animationRefs.current).forEach(tween => tween.pause());
    setIsAnimating(false);
  };

  const resetAnimation = () => {
    Object.entries(vehicleMarkersRef.current).forEach(([idx, marker]) => {
      const data = animationDataRef.current[idx];
      if (data?.points.length) marker.setLngLat(data.points[0]);
    });
    pauseAnimation();
  };

  return (
    <div className="App">
      <div className="controls">
        <h2>VRP Visualizer</h2>
        <label>Agentes:</label>
        <select value={numAgents} onChange={e => setNumAgents(Number(e.target.value))}>
          <option value={1}>1</option>
          <option value={2}>2</option>
          <option value={3}>3</option>
        </select>
        <h3>Seleccionar Ciudades:</h3>
        {citiesList.map((city, i) => (
          <div key={i}>
            <input
              type="checkbox"
              checked={selectedCities.includes(city)}
              disabled={city.isDepot}
              onChange={() =>
                setSelectedCities(selectedCities.includes(city)
                  ? selectedCities.filter(c => c !== city)
                  : [...selectedCities, city])
              }
            />
            {city.name}
          </div>
        ))}
        <button onClick={() => {
          clearRoute();
          setTimeout(() => optimizeRoute(), 100);
        }} disabled={isLoading}>
          {isLoading ? 'Calculando...' : 'Calcular Ruta'}
        </button>
        <button onClick={clearRoute}>Borrar Ruta</button>
        {route && (
          <>
            <h4>Resultado:</h4>
            {route.routes.map((r, i) => (
              <div key={i}>
                <strong>Vehículo {i + 1}:</strong> {r.route.join(' → ')}<br />
                Distancia: {r.distance.toFixed(2)} km
              </div>
            ))}
            <p><strong>Total:</strong> {route.total_distance.toFixed(2)} km</p>
          </>
        )}
        <div className="animation-controls">
          <button onClick={startAnimation} disabled={isAnimating}>Iniciar Animación</button>
          <button onClick={pauseAnimation}>Pausar</button>
          <button onClick={resetAnimation}>Reset</button>
          <label>
            Velocidad:
            <input
              type="range"
              min="1"
              max="100"
              step="10"
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
            />
          </label>
        </div>
      </div>
      <div ref={mapContainer} className="map-container" />
    </div>
  );
}

export default App;