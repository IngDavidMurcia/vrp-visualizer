import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './App.css';
import './mapPopupOverrides.css';

function App() {
  const [route, setRoute] = useState(null);
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [mapInitialized, setMapInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [numVehicles, setNumVehicles] = useState(1);

  // Coordenadas de Colombia
  const originalCoordinates = [
    [4.5709, -74.2973],  // Depósito (Bogotá centro)
    [6.2318, -75.5636],  // Medellín
    [4.639, -74.0817],   // Bogotá (otro punto)
    [4.4353, -75.2110]   // Ibagué
  ];

  // Función para calcular distancia
  const calculateDistance = (coord1, coord2) => {
    const [lat1, lon1] = coord1;
    const [lat2, lon2] = coord2;
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  const clearRoute = () => {
    setRoute(null);
    removeRouteLayers();
  };

  const removeRouteLayers = () => {
    try {
      // Limpiar todas las capas de vehículos
      const vehicleIds = Array.from({length: numVehicles}, (_, i) => i);
      vehicleIds.forEach(vehicleId => {
        const sourceId = `route-${vehicleId}`;
        const layerIds = [
          `route-line-${vehicleId}`,
          `route-points-${vehicleId}`,
          `route-labels-${vehicleId}`
        ];
        
        layerIds.forEach(layerId => {
          if (map.current.getLayer(layerId)) map.current.removeLayer(layerId);
        });
        
        if (map.current.getSource(sourceId)) map.current.removeSource(sourceId);
      });
    } catch (error) {
      console.log("Error limpiando capas:", error.message);
    }
  };

  const addRouteTooltips = () => {
    if (!map.current || !route) return;

    // Tooltip para la línea de ruta
    map.current.on('mouseenter', 'route-line', () => {
      map.current.getCanvas().style.cursor = 'pointer';
    });

    map.current.on('mouseleave', 'route-line', () => {
      map.current.getCanvas().style.cursor = '';
    });

    // Tooltip para puntos
    map.current.on('click', 'route-points', (e) => {
      const coordinates = e.features[0].geometry.coordinates.slice();
      const description = e.features[0].properties.name;
      
      // Solo calcular distancia si es un punto intermedio
      let distanceInfo = '';
      if (description.includes('Punto')) {
        const pointIndex = parseInt(description.split(' ')[1]);
        const currentRoute = route.routes?.find(r => r.route.includes(pointIndex)) || route;
        const currentRouteIndex = currentRoute.route.indexOf(pointIndex);
        
        if (currentRouteIndex < currentRoute.route.length - 1) {
          const nextPointIndex = currentRoute.route[currentRouteIndex + 1];
          const [nextLat, nextLon] = originalCoordinates[nextPointIndex];
          const dist = calculateDistance(
            [coordinates[1], coordinates[0]],
            [nextLat, nextLon]
          );
          distanceInfo = `<p>Distancia al siguiente punto: ${dist.toFixed(2)} km</p>`;
        }
      }

      new maplibregl.Popup()
        .setLngLat(coordinates)
        .setHTML(`
          <div class="route-popup">
            <h4>${description}</h4>
            <p>Coordenadas: ${coordinates[1].toFixed(4)}, ${coordinates[0].toFixed(4)}</p>
            ${description.includes('Punto') ? 
              `<p>Secuencia: ${route.route.indexOf(parseInt(description.split(' ')[1])) + 1}</p>` : ''}
            ${distanceInfo}
            <div class="popup-footer">
              <button class="close-popup" onclick="this.closest('.mapboxgl-popup').remove()">Cerrar</button>
            </div>
          </div>
        `)
        .addTo(map.current);
    });
  };

  // Inicialización del mapa
  useEffect(() => {
    if (!mapContainer.current) return;

    const mapTilerKey = 'wNHJXUkCv4sp6GAPuvsq';
    try {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: `https://api.maptiler.com/maps/streets/style.json?key=${mapTilerKey}`,
        center: [-74.2973, 4.5709],
        zoom: 4
      });

      map.current.on('load', () => {
        console.log('Mapa cargado correctamente');
        setMapInitialized(true);
        
        // Añadir marcadores iniciales
        originalCoordinates.forEach(([lat, lon], i) => {
          const markerElement = document.createElement('div');
          markerElement.className = 'custom-marker';
          const markerText = i === 0 ? 'A' : `P${i}`;

          markerElement.innerHTML = `
            <div class="marker-container">
              <div class="marker-pin" style="background: ${i === 0 ? '#FFA500' : '#3FB1CE'}"></div>
              <div class="marker-label">${markerText}</div>
            </div>
          `;

          new maplibregl.Marker({element: markerElement})
            .setLngLat([lon, lat])
            .setPopup(new maplibregl.Popup({ 
              closeButton: true,
              className: 'custom-popup',
              maxWidth: '300px'
            }).setHTML(`
              <div class="popup-content">
                <strong>${i === 0 ? 'Centro de Acopio' : `Punto ${i}`}</strong>
                <div>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}</div>
              </div>
            `))
            .addTo(map.current);
        });
      });

      map.current.on('error', (e) => {
        console.error('Error en el mapa:', e.error);
      });

      map.current.addControl(new maplibregl.NavigationControl(), 'top-right');
      map.current.addControl(
        new maplibregl.ScaleControl({
          maxWidth: 200,
          unit: 'metric'
        }),
        'bottom-left'
      );

      if (map.current) {
        import('./mapPopupOverrides.css');
      }

    } catch (error) {
      console.error('Error al crear el mapa:', error);
    }

    return () => {
      if (map.current) map.current.remove();
    };
  }, []);

  // Dibujar rutas para múltiples vehículos
  useEffect(() => {
    if (!mapInitialized || !route || !map.current) return;

    const colors = ['#4CAF50', '#2196F3', '#FF5722', '#9C27B0', '#FFC107'];
    
    // Manejar tanto respuesta antigua como nueva
    const routesToDraw = route.routes || [{
      vehicle_id: 0,
      route: route.route || [],
      distance: route.distance || 0
    }];

    routesToDraw.forEach((vehicleRoute, idx) => {
      const color = colors[idx % colors.length];
      
      // 1. Convertir índices a coordenadas
      const routePoints = vehicleRoute.route.map(index => {
        const [lat, lon] = originalCoordinates[index];
        return [lon, lat];
      });

      // 2. Añadir depósito al inicio y final
      const [depotLat, depoLon] = originalCoordinates[0];
      const fullRoute = [
        [depoLon, depotLat],
        ...routePoints,
        [depoLon, depotLat]
      ];

      // 3. Crear GeoJSON
      const geoJson = {
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: {
              type: "LineString",
              coordinates: fullRoute
            },
            properties: {
              vehicle_id: vehicleRoute.vehicle_id || idx
            }
          },
          ...fullRoute.map((coord, i) => ({
            type: "Feature",
            geometry: {
              type: "Point",
              coordinates: coord
            },
            properties: {
              name: i === 0 ? "Depósito (Inicio)" : 
                   i === fullRoute.length - 1 ? "Depósito (Fin)" :
                   `Punto ${vehicleRoute.route[i-1]}`
            }
          }))
        ]
      };

      // 4. Añadir al mapa
      const sourceId = `route-${vehicleRoute.vehicle_id || idx}`;
      map.current.addSource(sourceId, {
        type: 'geojson',
        data: geoJson
      });

      // Línea de ruta
      map.current.addLayer({
        id: `route-line-${vehicleRoute.vehicle_id || idx}`,
        type: 'line',
        source: sourceId,
        filter: ['==', ['geometry-type'], 'LineString'],
        paint: {
          'line-color': color,
          'line-width': 4,
          'line-opacity': 0
        }
      });

      // Puntos de ruta
      map.current.addLayer({
        id: `route-points-${vehicleRoute.vehicle_id || idx}`,
        type: 'circle',
        source: sourceId,
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-radius': [
            'case',
            ['any',
              ['==', ['get', 'name'], 'Depósito (Inicio)'],
              ['==', ['get', 'name'], 'Depósito (Fin)']
            ], 8, 6
          ],
          'circle-color': [
            'case',
            ['==', ['get', 'name'], 'Depósito (Inicio)'], '#FF5722',
            ['==', ['get', 'name'], 'Depósito (Fin)'], '#FF5722',
            color
          ],
          'circle-stroke-width': 2,
          'circle-stroke-color': '#fff',
          'circle-opacity': 0
        }
      });

      // Animación
      setTimeout(() => {
        map.current.setPaintProperty(
          `route-line-${vehicleRoute.vehicle_id || idx}`,
          'line-opacity',
          0.8,
          { duration: 1000 }
        );
        map.current.setPaintProperty(
          `route-points-${vehicleRoute.vehicle_id || idx}`,
          'circle-opacity',
          1,
          { duration: 800 }
        );
      }, idx * 300);
    });

    // Ajustar vista
    const bounds = new maplibregl.LngLatBounds();
    routesToDraw.forEach(vehicleRoute => {
      vehicleRoute.route.forEach(index => {
        const [lat, lon] = originalCoordinates[index];
        bounds.extend([lon, lat]);
      });
    });
    map.current.fitBounds(bounds, { padding: 50, maxZoom: 12 });

    // Añadir tooltips
    addRouteTooltips();

  }, [route, mapInitialized]);

  const optimizeRoute = async () => {
    setIsLoading(true);
    clearRoute();
    
    try {
          // Redondear coordenadas antes de enviar
      const roundedCoords = originalCoordinates.map(coord => [
        parseFloat(coord[0].toFixed(6)),
       parseFloat(coord[1].toFixed(6))
      ]);

      const response = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          points: roundedCoords,
          num_vehicles: numVehicles
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Error HTTP: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Respuesta del backend:", data);
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
          <h3>Ruta optimizada</h3>
          <div className="route-summary">
            <div>
              <span className="label">Vehículos:</span>
              <span className="value">{route.routes?.length || 1}</span>
            </div>
            <div>
              <span className="label">Distancia total:</span>
              <span className="value">
              {(route.total_distance ? route.total_distance : route.distance / 1000).toFixed(2)} km
              </span>
            </div>
          </div>
          
          {(route.routes || [route]).map((vehicleRoute, i) => (
            <div key={i} className="vehicle-route">
              <h4>Vehículo {i + 1}</h4>
              <p>
                <strong>Secuencia:</strong> Depósito → 
                {vehicleRoute.route.map(index => ` Punto ${index}`).join(" → ")} → Depósito
              </p>
              <p><strong>Distancia:</strong> {(vehicleRoute.distance / 1000).toFixed(2)} km</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;