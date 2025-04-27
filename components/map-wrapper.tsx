"use client"

import dynamic from "next/dynamic"
import { Loader2 } from "lucide-react"

// Dynamically import the map component with ssr: false
const WildfireMap = dynamic(() => import("@/components/wildfire-map").then((mod) => mod.WildfireMap), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center bg-gray-100 h-full w-full">
      <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
    </div>
  ),
})

export function MapWrapper() {
  return <WildfireMap className="h-full" />
}
