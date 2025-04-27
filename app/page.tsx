import { WildfireHeader } from "@/components/wildfire-header"
import { AnalysisPanel } from "@/components/analysis-panel"
import { MapWrapper } from "@/components/map-wrapper"
import MapClient from "@/components/map-client"

export default function Home() {
  return (
    <main className="flex h-screen">
      <div className="relative flex-1 transition-all duration-300 ease-in-out">
        <MapClient />
      </div>
      <div className="w-[30%] border-l hover:w-[50%] transition-all duration-300 ease-in-out z-10 bg-background shadow-lg">
        <AnalysisPanel />
      </div>
    </main>
  )
}
