import React, { useRef, useEffect } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

export default function Map({ route }) {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (!map.current) {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: 'https://demotiles.maplibre.org/style.json', // Estilo básico
        center: [-96, 37.8], // Centro inicial de EE.UU.
        zoom: 3
      });
      
      // Añadir controles de navegación
      map.current.addControl(new maplibregl.NavigationControl());
    }

    // Limpieza al desmontar
    return () => map.current?.remove();
  }, []);

  return (
    <div 
      ref={mapContainer} 
      style={{ 
        width: '100%', 
        height: '500px',
        borderRadius: '8px',
        border: '1px solid #ccc'
      }} 
    />
  );
}