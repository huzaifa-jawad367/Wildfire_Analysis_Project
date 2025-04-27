import { create } from "zustand"
import { analyzeWildfire } from "./api"

interface Location {
  latitude: number
  longitude: number
}

interface AnalysisParams {
  latitude: number
  longitude: number
  preFireDate: string
  postFireDate: string
}

interface BurnAreaFeature {
  type: 'Feature'
  geometry: {
    type: 'Polygon'
    coordinates: [number, number][][]
  }
  properties: {
    severity: string
    area_approx: number
    is_hotspot?: boolean
  }
}

interface BurnAreaCollection {
  type: 'FeatureCollection'
  features: BurnAreaFeature[]
}

interface HighSeverityCoordinate {
  latitude: number
  longitude: number
  severity: 'extreme' | 'very_high'
}

interface AnalysisResults {
  location: Location
  preFireDate: string
  postFireDate: string
  dataSource: string
  totalBurnedArea: number
  burnSeverityStats: {
    low: number
    moderate: number
    high: number
    veryHigh: number
    extreme: number
  }
  nbrStats: {
    preFireAvg: number
    postFireAvg: number
    dNBRAvg: number
    dNBRMax: number
  }
  images: {
    preFireTrueColor: string
    postFireTrueColor: string
    preFireNBR: string
    postFireNBR: string
    dNBR: string
    burnSeverity: string
    burnSeverityWithHotspot: string
    burnSeverityLegend: string
  }
  burnSeverityPolygons?: BurnAreaCollection
  highSeverityCoordinates: BurnAreaCollection
  status?: string
  timestamp?: string
  showHotspot?: boolean
}

interface WildfireStore {
  selectedLocation: Location | null
  setSelectedLocation: (location: Location) => void

  analysisResults: AnalysisResults | null
  burnSeverityPolygons: BurnAreaCollection | null

  isLoading: boolean
  error: string | null

  visibleLayers: {
    burnSeverity: boolean
    hotspot: boolean
  }
  toggleLayerVisibility: (layer: keyof WildfireStore["visibleLayers"]) => void

  runAnalysis: (params: AnalysisParams) => Promise<void>
  clearAnalysis: () => void
}

export const useWildfireStore = create<WildfireStore>((set, get) => ({
  selectedLocation: null,
  setSelectedLocation: (location) => {
    set({
      selectedLocation: location,
      // Clear previous analysis results when a new location is selected
      analysisResults: null,
      burnSeverityPolygons: null,
      error: null,
    })
  },

  analysisResults: null,
  burnSeverityPolygons: null,

  isLoading: false,
  error: null,

  visibleLayers: {
    burnSeverity: true,
    hotspot: true,
  },
  toggleLayerVisibility: (layer) => {
    set((state) => ({
      visibleLayers: {
        ...state.visibleLayers,
        [layer]: !state.visibleLayers[layer],
      },
    }))
  },

  runAnalysis: async (params) => {
    set({ isLoading: true, error: null })

    try {
      console.log("Running analysis with params:", params)

      // Call the API to run the Python script
      const results = await analyzeWildfire(params)

      // Debug logging
      console.log("Analysis results received:", {
        hasResults: !!results,
        hasPolygons: !!results.burnSeverityPolygons,
        polygonFeatures: results.burnSeverityPolygons?.features?.length || 0,
        highSeverityCoords: results.highSeverityCoordinates?.features?.length || 0
      });

      // Use the polygons directly from Python script
      const geojsonData = results.burnSeverityPolygons || null

      set({
        analysisResults: results,
        burnSeverityPolygons: geojsonData,
        isLoading: false,
      })
    } catch (error) {
      console.error("Error running analysis:", error)
      set({
        error: error instanceof Error ? error.message : "An unknown error occurred",
        isLoading: false,
      })
    }
  },

  clearAnalysis: () => {
    set({
      analysisResults: null,
      burnSeverityPolygons: null,
      error: null,
    })
  },
}))
