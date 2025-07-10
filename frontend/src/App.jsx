import React, { useState } from 'react';
import Map from './components/Map';
import './App.css';

function App() {
  const [route, setRoute] = useState(null);

  const optimizeRoute = async () => {
    const response = await fetch('http://localhost:8000/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: [
          [40.71, -74.01], // NYC
          [34.05, -118.24], // LA
          [41.88, -87.62]   // Chicago
        ]
      })
    });
    setRoute(await response.json());
  };

  return (
    <div className="app-container">
      <h1>Optimizador de Rutas VRP</h1>
      <button onClick={optimizeRoute} className="optimize-btn">
        Calcular Ruta Óptima
      </button>
      <div className="map-container">
        <Map route={route} />
      </div>
      {route && (
        <div className="route-info">
          <h3>Resultados:</h3>
          <pre>{JSON.stringify(route, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
