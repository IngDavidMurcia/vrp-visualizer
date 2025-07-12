import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './App.css';
import './mapPopupOverrides.css';

// Listado predefinido de ciudades (7 Colombia + 3 EEUU)
const PREDEFINED_CITIES = [
  { name: "Bogotá (Centro)", coords: [4.5709, -74.2973], isDepot: true },
  { name: "Medellín", coords: [6.2318, -75.5636] },
  { name: "Cali", coords: [3.4516, -76.5320] },
  { name: "Barranquilla", coords: [10.9639, -74.7964] },
  { name: "Cartagena", coords: [10.3910, -75.4794] },
  { name: "Bucaramanga", coords: [7.1193, -73.1227] },
  { name: "Pereira", coords: [4.8143, -75.6946] },
  { name: "New York", coords: [40.7128, -74.0060] },
  { name: "Miami", coords: [25.7617, -80.1918] },
  { name: "Los Angeles", coords: [34.0522, -118.2437] }
];

function App() {
  const [route, setRoute] = useState(null);
  const [selectedCities, setSelectedCities] = useState(
    PREDEFINED_CITIES.filter(c => c.isDepot || Math.random() < 0.3).slice(0, 3)
  );
  const [allCities, setAllCities] = useState(PREDEFINED_CITIES);
  const [numVehicles, setNumVehicles] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [mapInitialized, setMapInitialized] = useState(false);
  const [activeMenu, setActiveMenu] = useState(null);
  const [newCity, setNewCity] = useState({ name: '', lat: '', lon: '' });
  const mapContainer = useRef(null);
  const map = useRef(null);

  // Funciones básicas
  const clearRoute = () => {
    setRoute(null);
    removeRouteLayers();
  };

  const removeRouteLayers = () => {
    try {
      const layers = ['route-line', 'route-points', 'route-labels'];
      layers.forEach(layer => {
        if (map.current?.getLayer(layer)) map.current.removeLayer(layer);
      });
      if (map.current?.getSource('route')) map.current.removeSource('route');
    } catch (error) {
      console.error("Error limpiando capas:", error);
    }
  };

  // Menú de opciones
  const renderMenu = () => {
    switch (activeMenu) {
      case 'select-cities':
        return (
          <div className="menu-container">
            <h3>Seleccionar Ciudades</h3>
            {allCities.map((city, i) => (
              <div key={i} className="city-item">
                <input
                  type="checkbox"
                  checked={selectedCities.some(c => c.name === city.name)}
                  onChange={() => toggleCitySelection(city)}
                />
                <span>{city.name}</span>
              </div>
            ))}
            <button onClick={() => setActiveMenu(null)}>Aplicar</button>
          </div>
        );
      case 'add-city':
        return (
          <div className="menu-container">
            <h3>Añadir Ciudad</h3>
            <input
              type="text"
              placeholder="Nombre"
              value={newCity.name}
              onChange={(e) => setNewCity({...newCity, name: e.target.value})}
            />
            <input
              type="number"
              placeholder="Latitud"
              step="0.0001"
              value={newCity.lat}
              onChange={(e) => setNewCity({...newCity, lat: e.target.value})}
            />
            <input
              type="number"
              placeholder="Longitud"
              step="0.0001"
              value={newCity.lon}
              onChange={(e) => setNewCity({...newCity, lon: e.target.value})}
            />
            <button onClick={handleAddCity}>Añadir</button>
            <button onClick={() => setActiveMenu(null)}>Cancelar</button>
          </div>
        );
      case 'remove-city':
        return (
          <div className="menu-container">
            <h3>Eliminar Ciudades</h3>
            {selectedCities.filter(c => !c.isDepot).map((city, i) => (
              <div key={i} className="city-item">
                <button onClick={() => removeCity(city)}>Eliminar</button>
                <span>{city.name}</span>
              </div>
            ))}
            <button onClick={() => setActiveMenu(null)}>Listo</button>
          </div>
        );
      case 'vehicles':
        return (
          <div className="menu-container">
            <h3>Seleccionar Vehículos</h3>
            {[1, 2, 3].map(num => (
              <div key={num} className="vehicle-option">
                <input
                  type="radio"
                  id={`vehicle-${num}`}
                  name="vehicles"
                  checked={numVehicles === num}
                  onChange={() => setNumVehicles(num)}
                />
                <label htmlFor={`vehicle-${num}`}>{num} Vehículo(s)</label>
              </div>
            ))}
            <button onClick={() => setActiveMenu(null)}>Aplicar</button>
          </div>
        );
      default:
        return null;
    }
  };

  // Funciones de manejo de ciudades
  const toggleCitySelection = (city) => {
    if (city.isDepot) return;
    setSelectedCities(prev => 
      prev.some(c => c.name === city.name)
        ? prev.filter(c => c.name !== city.name)
        : [...prev, city]
    );
  };

  const handleAddCity = () => {
    if (!newCity.name || !newCity.lat || !newCity.lon) return;
    const lat = parseFloat(newCity.lat);
    const lon = parseFloat(newCity.lon);
    
    if (isNaN(lat) || isNaN(lon)) {
      alert("Coordenadas inválidas");
      return;
    }

    const newCityObj = {
      name: newCity.name,
      coords: [lat, lon]
    };

    setAllCities([...allCities, newCityObj]);
    setSelectedCities([...selectedCities, newCityObj]);
    setNewCity({ name: '', lat: '', lon: '' });
    setActiveMenu(null);
  };

  const removeCity = (city) => {
    setSelectedCities(prev => prev.filter(c => c.name !== city.name));
  };

  // Optimización de ruta
  const optimizeRoute = async () => {
    if (selectedCities.length < 3) {
      alert("Seleccione al menos 3 ciudades además del depósito");
      return;
    }

    setIsLoading(true);
    clearRoute();

    try {
      // Preparar datos para enviar
      const requestData = {
        points: selectedCities.map(city => city.coords),
        num_vehicles: numVehicles
      };
  
      console.log("Enviando datos:", requestData); // Para depuración
  
      const response = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error del backend:", errorData); // Log detallado
        throw new Error(errorData.detail || `Error HTTP: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Respuesta recibida:", data); // Para depuración
      setRoute(data);
      
    } catch (error) {
      console.error("Error completo:", error);
      alert(`Error al calcular ruta: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Inicialización del mapa y efectos
  useEffect(() => {
    if (!mapContainer.current) return;

    const mapTilerKey = 'wNHJXUkCv4sp6GAPuvsq';
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: `https://api.maptiler.com/maps/streets/style.json?key=${mapTilerKey}`,
      center: [-74.2973, 4.5709],
      zoom: 4
    });

    map.current.on('load', () => {
      setMapInitialized(true);
      updateMapMarkers();
    });

    return () => map.current?.remove();
  }, []);

  useEffect(() => {
    if (mapInitialized) {
      updateMapMarkers();
    }
  }, [selectedCities, mapInitialized]);

  useEffect(() => {
    if (mapInitialized && route) {
      drawRoute();
    }
  }, [route, mapInitialized]);

  // Funciones de mapa
  const updateMapMarkers = () => {
    if (!map.current) return;

    // Limpiar marcadores existentes
    const markers = document.querySelectorAll('.mapboxgl-marker');
    markers.forEach(marker => marker.remove());

    // Añadir nuevos marcadores
    selectedCities.forEach((city, i) => {
      const markerElement = document.createElement('div');
      markerElement.className = 'custom-marker';
      markerElement.innerHTML = `
        <div class="marker-container">
          <div class="marker-pin" style="background: ${city.isDepot ? '#FFA500' : '#3FB1CE'}"></div>
          <div class="marker-label">${city.isDepot ? 'A' : `P${i}`}</div>
        </div>
      `;

      new maplibregl.Marker({ element: markerElement })
        .setLngLat([city.coords[1], city.coords[0]])
        .setPopup(new maplibregl.Popup().setHTML(`
          <div class="popup-content">
            <strong>${city.name}</strong>
            <div>Lat: ${city.coords[0].toFixed(4)}, Lon: ${city.coords[1].toFixed(4)}</div>
          </div>
        `))
        .addTo(map.current);
    });
  };

  const drawRoute = () => {
    if (!map.current || !route) return;
    removeRouteLayers();

    const colors = ['#4CAF50', '#2196F3', '#FF5722'];
    const routesToDraw = route.routes || [route];

    routesToDraw.forEach((vehicleRoute, idx) => {
      const color = colors[idx % colors.length];
      const routeCoords = vehicleRoute.route.map(index => {
        const city = selectedCities[index];
        return [city.coords[1], city.coords[0]];
      });

      // Añadir depósito al inicio y final
      const depot = selectedCities.find(c => c.isDepot);
      const fullRoute = [
        [depot.coords[1], depot.coords[0]],
        ...routeCoords,
        [depot.coords[1], depot.coords[0]]
      ];

      // Añadir fuente y capas
      map.current.addSource(`route-${idx}`, {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [{
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: fullRoute
            }
          }]
        }
      });

      map.current.addLayer({
        id: `route-line-${idx}`,
        type: 'line',
        source: `route-${idx}`,
        paint: {
          'line-color': color,
          'line-width': 4,
          'line-opacity': 0.8
        }
      });
    });
  };

  return (
    <div className="app-container">
      <h1>Optimizador IA de Rutas (VRP)</h1>
      
      <div className="controls">
        <button onClick={() => setActiveMenu('select-cities')}>Elegir Ciudades</button>
        <button onClick={() => setActiveMenu('add-city')}>Añadir Ciudad</button>
        <button onClick={() => setActiveMenu('remove-city')}>Borrar Ciudades</button>
        <button onClick={() => setActiveMenu('vehicles')}>Vehículos: {numVehicles}</button>
        <button onClick={optimizeRoute} disabled={isLoading}>
          {isLoading ? 'Calculando...' : 'Calcular Ruta'}
        </button>
        <button onClick={clearRoute}>Borrar Ruta</button>
      </div>

      {renderMenu()}

      <div 
        ref={mapContainer} 
        className="map-container" 
        style={{ border: '2px solid #ccc' }}
      />

      {route && (
        <div className="route-info">
          <h3>Ruta Optimizada</h3>
          <div className="summary">
            <p><strong>Total ciudades:</strong> {selectedCities.length - 1}</p>
            <p><strong>Vehículos utilizados:</strong> {route.routes?.length || 1}</p>
            <p><strong>Distancia total:</strong> {(route.total_distance || route.distance / 1000).toFixed(2)} km</p>
          </div>
          
          {(route.routes || [route]).map((vehicleRoute, i) => (
            <div key={i} className="vehicle-route">
              <h4>Vehículo {i + 1}</h4>
              <p><strong>Recorrido:</strong> Depósito → {
                vehicleRoute.route.map(index => selectedCities[index].name).join(" → ")
              } → Depósito</p>
              <p><strong>Distancia:</strong> {(vehicleRoute.distance / 1000).toFixed(2)} km</p>
              <p><strong>Ciudades visitadas:</strong> {vehicleRoute.route.length}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;