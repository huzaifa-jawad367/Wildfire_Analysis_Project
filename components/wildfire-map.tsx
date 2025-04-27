"use client"

import { useState, useEffect } from "react"
import { Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import dynamic from "next/dynamic"

// Dynamically import the map component with ssr: false
const MapClient = dynamic(() => import("@/components/map-client"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center bg-gray-100 h-full w-full">
      <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
    </div>
  ),
})

interface WildfireMapProps {
  className?: string
}

export function WildfireMap({ className }: WildfireMapProps) {
  const [isMounted, setIsMounted] = useState(false)

  // Only render the map on the client side
  useEffect(() => {
    setIsMounted(true)
  }, [])

  return (
    <div className={cn("relative", className)}>
      {isMounted ? (
        <MapClient />
      ) : (
        <div className="flex items-center justify-center bg-gray-100 h-full w-full">
          <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
        </div>
      )}
    </div>
  )
}
