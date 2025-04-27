// Simplify the API client to directly call the Python script
export interface AnalysisParams {
  latitude: number
  longitude: number
  preFireDate: string
  postFireDate: string
  bufferKm?: number
  showHotspot?: boolean
}

export interface HighSeverityCoordinate {
  latitude: number
  longitude: number
  severity: 'extreme' | 'very_high'
  is_hotspot?: boolean
}

export interface BurnAreaFeature {
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

export interface BurnAreaCollection {
  type: 'FeatureCollection'
  features: BurnAreaFeature[]
  hotspot_window?: [number, number, number, number]
}

export interface AnalysisResults {
  location: {
    latitude: number
    longitude: number
  }
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

// Direct API client to call the Python script
export async function analyzeWildfire(params: AnalysisParams): Promise<AnalysisResults> {
  try {
    console.log("Sending analysis request to API:", params)

    const response = await fetch("/api/analyze-wildfire", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    })

    let errorMessage = ""
    let responseData = null

    try {
      // Always try to get the response data, whether it's an error or success
      responseData = await response.json()
    } catch (parseError) {
      console.error("Error parsing response:", parseError)
      // If we can't parse JSON, try to get text
      try {
        const textResponse = await response.text()
        errorMessage = `Server returned invalid JSON. Response: ${textResponse.substring(0, 200)}...`
      } catch (textError) {
        errorMessage = `Server returned invalid response: ${response.statusText}`
      }
      throw new Error(errorMessage)
    }

    // Now we have the parsed response data
    if (!response.ok) {
      errorMessage = "Analysis failed"

      if (responseData) {
        console.error("Error data from server:", responseData)

        if (responseData.error) {
          errorMessage = responseData.error
        }

        // Add details if available
        if (responseData.details) {
          errorMessage += `\n\nDetails: ${
            typeof responseData.details === "string" 
              ? responseData.details 
              : JSON.stringify(responseData.details, null, 2)
          }`
        }

        // Add stdout/stderr if available
        if (responseData.stdout) {
          errorMessage += `\n\nOutput: ${responseData.stdout}`
        }
        if (responseData.stderr) {
          errorMessage += `\n\nError Output: ${responseData.stderr}`
        }

        // Add command if available
        if (responseData.command) {
          errorMessage += `\n\nCommand: ${responseData.command}`
        }
      }

      throw new Error(errorMessage)
    }

    // Success case - we have valid results
    console.log("Received analysis results:", responseData)
    return responseData as AnalysisResults
  } catch (error) {
    console.error("Error analyzing wildfire:", error)
    throw error // Re-throw the error to be handled by the caller
  }
}
