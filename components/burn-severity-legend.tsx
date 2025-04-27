import { cn } from "@/lib/utils"
import { useWildfireStore } from "@/lib/store"

interface BurnSeverityLegendProps {
  className?: string
}

export function BurnSeverityLegend({ className }: BurnSeverityLegendProps) {
  const { visibleLayers } = useWildfireStore()
  
  const severityClasses = [
    ...(visibleLayers.hotspot ? [
      { label: "Low", color: "#69B34C" },
      { label: "Moderate", color: "#FAB733" },
      { label: "High", color: "#FF8E15" },
      { label: "Very High", color: "#FF4E11" },
      { label: "Extreme", color: "#FF0D0D" },
      { label: "Hotspot", color: "#800080" }
    ] : [
      { label: "Hotspot", color: "#800080" }
    ])
  ]

  return (
    <div className={cn("flex flex-wrap items-center justify-center gap-2 p-2 bg-white rounded-md shadow-sm", className)}>
      {severityClasses.map((severity) => (
        <div key={severity.label} className="flex items-center gap-1">
          <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: severity.color }} />
          <span className="text-xs">{severity.label}</span>
        </div>
      ))}
    </div>
  )
}
