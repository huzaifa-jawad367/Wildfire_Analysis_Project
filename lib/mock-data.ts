export const mockAnalysisResults = {
  location: {
    latitude: 37.7749,
    longitude: -122.4194,
  },
  preFireDate: "2023-06-01",
  postFireDate: "2023-08-01",
  dataSource: "Sentinel-2",
  totalBurnedArea: 42.75,
  burnSeverityStats: {
    low: 12.5,
    moderate: 15.3,
    high: 8.7,
    veryHigh: 4.2,
    extreme: 2.05,
  },
  nbrStats: {
    preFireAvg: 0.412,
    postFireAvg: 0.187,
    dNBRAvg: 0.225,
    dNBRMax: 0.687,
  },
  images: {
    preFireImage: "images/pre_fire.png",
    postFireImage: "images/post_fire.png",
    preFireNBR: "images/pre_fire_nbr.png",
    postFireNBR: "images/post_fire_nbr.png",
    dNBR: "images/dnbr.png",
    burnSeverity: "images/burn_severity.png",
    burnSeverityLegend: "images/burn_severity_legend.png",
  },
}
