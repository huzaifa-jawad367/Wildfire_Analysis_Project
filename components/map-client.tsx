"use client"

import { useEffect } from "react"
import dynamic from "next/dynamic"
import { useWildfireStore } from "@/lib/store"
import { Button } from "@/components/ui/button"
import { Eye, EyeOff, Loader2 } from "lucide-react"
import { BurnSeverityLegend } from "@/components/burn-severity-legend"
import { cn } from "@/lib/utils"
import type { MapContainer as MapContainerType, TileLayer as TileLayerType, CircleMarker as CircleMarkerType, Marker as MarkerType, Popup as PopupType, GeoJSON as GeoJSONType } from "react-leaflet"
import { useMap } from 'react-leaflet'
import type { BurnAreaFeature as BaseBurnAreaFeature, BurnAreaCollection } from "@/lib/api"

// Extend the base feature type to include hotspot property
interface BurnAreaFeature extends Omit<BaseBurnAreaFeature, 'properties'> {
  properties: {
    severity: string
    area_approx: number
    is_hotspot?: boolean
  }
}

// Dynamically import Leaflet components with no SSR
const MapContainer = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
) as typeof MapContainerType

const TileLayer = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
) as typeof TileLayerType

const CircleMarker = dynamic(
  () => import("react-leaflet").then((mod) => mod.CircleMarker),
  { ssr: false }
) as typeof CircleMarkerType

const Marker = dynamic(
  () => import("react-leaflet").then((mod) => mod.Marker),
  { ssr: false }
) as typeof MarkerType

const Popup = dynamic(
  () => import("react-leaflet").then((mod) => mod.Popup),
  { ssr: false }
) as typeof PopupType

const GeoJSON = dynamic(
  () => import("react-leaflet").then((mod) => mod.GeoJSON),
  { ssr: false }
) as typeof GeoJSONType

// Import Leaflet CSS
import "leaflet/dist/leaflet.css"

// This component handles map clicks
function MapClickHandler() {
  const { isLoading, setSelectedLocation } = useWildfireStore()
  const map = useMap()

  useEffect(() => {
    if (!map) return

    const handleClick = (e: { latlng: { lat: number; lng: number } }) => {
      if (isLoading) return

      console.log("Map clicked at:", e.latlng)
      const { lat, lng } = e.latlng
      setSelectedLocation({
        latitude: lat,
        longitude: lng,
      })
    }

    map.on("click", handleClick)

    return () => {
      map.off("click", handleClick)
    }
  }, [map, isLoading, setSelectedLocation])

  return null
}

export default function MapClient() {
  const { selectedLocation, burnSeverityPolygons, isLoading, toggleLayerVisibility, visibleLayers, analysisResults } = useWildfireStore()

  // Enhanced debug logging
  useEffect(() => {
    if (analysisResults) {
      console.log('Analysis Results:', {
        hasResults: !!analysisResults,
        hasHighSeverityCoords: !!analysisResults.highSeverityCoordinates,
        numCoords: analysisResults.highSeverityCoordinates?.features?.length || 0,
        visibleLayers,
        sampleCoord: analysisResults.highSeverityCoordinates?.features?.[0]
      });
    }
  }, [analysisResults, visibleLayers]);

  // Fix Leaflet icon issues
  useEffect(() => {
    const initializeLeaflet = async () => {
      const L = (await import('leaflet')).default
      delete (L.Icon.Default.prototype as any)._getIconUrl
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
      })
    }
    initializeLeaflet()
  }, [])

  if (!MapContainer) {
    return <div>Loading map...</div>
  }

  return (
    <>
      <MapContainer
        center={selectedLocation ? [selectedLocation.latitude, selectedLocation.longitude] : [37.7749, -122.4194]}
        zoom={10}
        className="h-full w-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapClickHandler />

        {selectedLocation && (
          <Marker position={[selectedLocation.latitude, selectedLocation.longitude]}>
            <Popup>
              Selected Location
              <br />
              {selectedLocation.latitude.toFixed(6)}, {selectedLocation.longitude.toFixed(6)}
            </Popup>
          </Marker>
        )}

        {/* Render burn severity polygons */}
        {visibleLayers.burnSeverity && burnSeverityPolygons && burnSeverityPolygons.features && burnSeverityPolygons.features.length > 0 && (
          <GeoJSON
            data={burnSeverityPolygons}
            style={(feature) => ({
              fillColor: getBurnSeverityColor(feature?.properties?.severity || 0),
              color: getBurnSeverityColor(feature?.properties?.severity || 0),
              weight: 1,
              opacity: 0.8,
              fillOpacity: 0.5,
            })}
            onEachFeature={(feature, layer) => {
              layer.bindPopup(`
                <strong>Burn Severity:</strong> ${getSeverityLabel(feature?.properties?.severity || 0)}
                ${feature?.properties?.area ? `<br><strong>Area:</strong> ${feature.properties.area.toFixed(2)} km²` : ''}
              `)
            }}
          />
        )}

        {/* Extreme severity polygons with hotspot highlighting */}
        {(() => {
          if (!visibleLayers.burnSeverity || !analysisResults?.highSeverityCoordinates?.features?.length) {
            return null;
          }
          
          return (
            <>
              {analysisResults.highSeverityCoordinates.features.map((feature: BurnAreaFeature, index: number) => {
                // Show only hotspots (purple) when hotspot layer is off
                // Show all (both red and purple) when hotspot layer is on
                if (!visibleLayers.hotspot && !feature.properties.is_hotspot) {
                  return null;
                }
                
                const color = feature.properties.is_hotspot ? '#800080' : '#FF0D0D';
                
                return (
                  <GeoJSON
                    key={`severity-${index}`}
                    data={feature}
                    style={{
                      fillColor: color,
                      color: color,
                      weight: 2,
                      opacity: 1,
                      fillOpacity: 0.5
                    }}
                  >
                    <Popup>
                      <strong>Severity:</strong> {feature.properties.is_hotspot ? 'Hotspot' : 'Extreme'}
                      <br />
                      <strong>Area:</strong> {feature.properties.area_approx.toFixed(2)} km²
                    </Popup>
                  </GeoJSON>
                );
              })}
            </>
          );
        })()}

        <MapCenterUpdater />
      </MapContainer>

      <div className="absolute top-4 right-4 z-[1000] flex flex-col gap-2">
        <div className="bg-white p-2 rounded-md shadow-md flex flex-col gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => toggleLayerVisibility("burnSeverity")}
            className={cn(
              "flex items-center gap-2",
              visibleLayers.burnSeverity ? "bg-primary text-primary-foreground" : "",
            )}
          >
            {visibleLayers.burnSeverity ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            Burn Severity
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => toggleLayerVisibility("hotspot")}
            className={cn(
              "flex items-center gap-2",
              visibleLayers.hotspot ? "bg-primary text-primary-foreground" : "",
            )}
          >
            {visibleLayers.hotspot ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            {visibleLayers.hotspot ? "Show All Severities" : "Show Hotspots Only"}
          </Button>
        </div>

        {burnSeverityPolygons && visibleLayers.burnSeverity && <BurnSeverityLegend />}
      </div>

      {isLoading && (
        <div className="absolute inset-0 bg-white/20 flex items-center justify-center z-[1000]">
          <div className="bg-white p-4 rounded-md shadow-md flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span>Analyzing wildfire data...</span>
          </div>
        </div>
      )}
    </>
  )
}

// Helper component to update map center when selectedLocation changes
function MapCenterUpdater() {
  const { selectedLocation } = useWildfireStore()
  const map = useMap()

  useEffect(() => {
    if (selectedLocation && map) {
      map.setView(
        [selectedLocation.latitude, selectedLocation.longitude],
        map.getZoom() === 1 ? 10 : map.getZoom() // Set zoom to 10 if it's the default zoom
      )
    }
  }, [selectedLocation, map])

  return null
}

function getBurnSeverityColor(severity: number): string {
  switch (severity) {
    case 1:
      return "#69B34C" // Low
    case 2:
      return "#FAB733" // Moderate
    case 3:
      return "#FF8E15" // High
    case 4:
      return "#FF4E11" // Very High
    case 5:
      return "#FF0D0D" // Extreme
    default:
      return "#CCCCCC"
  }
}

function getSeverityLabel(severity: number): string {
  switch (severity) {
    case 1:
      return "Low"
    case 2:
      return "Moderate"
    case 3:
      return "High"
    case 4:
      return "Very High"
    case 5:
      return "Extreme"
    default:
      return "Unknown"
  }
}
