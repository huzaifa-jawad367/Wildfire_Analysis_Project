// Mock data for development/demo purposes
export function getMockAnalysisResults(params: any): any {
  // Generate random but realistic data based on the location
  const totalBurnedArea = 30 + Math.random() * 30 // 30-60 km²

  // Distribute the total area across severity classes
  const lowArea = totalBurnedArea * (0.2 + Math.random() * 0.2)
  const moderateArea = totalBurnedArea * (0.2 + Math.random() * 0.2)
  const highArea = totalBurnedArea * (0.15 + Math.random() * 0.15)
  const veryHighArea = totalBurnedArea * (0.1 + Math.random() * 0.1)
  const extremeArea = totalBurnedArea - lowArea - moderateArea - highArea - veryHighArea

  const burnSeverityStats = {
    low: lowArea,
    moderate: moderateArea,
    high: highArea,
    veryHigh: veryHighArea,
    extreme: extremeArea,
  }

  // Create the mock results
  const results = {
    location: {
      latitude: params.latitude,
      longitude: params.longitude,
    },
    preFireDate: params.preFireDate,
    postFireDate: params.postFireDate,
    dataSource: "Sentinel-2 (Mock Data)",
    totalBurnedArea: totalBurnedArea,
    burnSeverityStats: burnSeverityStats,
    nbrStats: {
      preFireAvg: 0.3 + Math.random() * 0.2,
      postFireAvg: 0.1 + Math.random() * 0.15,
      dNBRAvg: 0.15 + Math.random() * 0.15,
      dNBRMax: 0.5 + Math.random() * 0.3,
    },
    images: {
      preFireTrueColor: "images/pre_fire.png",
      postFireTrueColor: "images/post_fire.png",
      preFireNBR: "images/pre_fire_nbr.png",
      postFireNBR: "images/post_fire_nbr.png",
      dNBR: "images/dnbr.png",
      burnSeverity: "images/burn_severity.png",
      burnSeverityLegend: "images/burn_severity_legend.png",
    },
    status: "completed",
    timestamp: new Date().toISOString(),
    burnSeverityPolygons: createBurnSeverityGeoJSON(params.latitude, params.longitude, burnSeverityStats),
  }

  return results
}

// Helper function to create GeoJSON data for the burn severity map
export function createBurnSeverityGeoJSON(lat: number, lng: number, areas: any) {
  const features = []
  const severityClasses = ["low", "moderate", "high", "veryHigh", "extreme"]
  const severityValues = [1, 2, 3, 4, 5]

  // Create a polygon for each severity class
  for (let i = 0; i < severityClasses.length; i++) {
    const severityClass = severityClasses[i]
    const area = areas[severityClass]

    if (area > 0) {
      // Calculate a radius based on the area (simplified approach)
      const radius = Math.sqrt(area) / 10

      // Create polygons with irregular shapes to simulate real burn areas
      const points = 20
      const coordinates: Array<Array<[number, number]>> = [[]]

      for (let j = 0; j < points; j++) {
        const angle = (j / points) * Math.PI * 2
        // Add some randomness to make the polygons look more natural
        const randomFactor = 0.7 + Math.random() * 0.6
        const x = lng + Math.cos(angle) * radius * randomFactor * (1 + i * 0.2)
        const y = lat + Math.sin(angle) * radius * randomFactor * (1 + i * 0.2)
        coordinates[0].push([x, y])
      }

      // Close the polygon
      coordinates[0].push(coordinates[0][0])

      features.push({
        type: "Feature",
        properties: {
          severity: severityValues[i],
          area: area,
          severityClass: severityClass,
        },
        geometry: {
          type: "Polygon",
          coordinates: coordinates,
        },
      })
    }
  }

  return {
    type: "FeatureCollection",
    features: features,
  }
}
