"use client"

import { useEffect, useRef } from "react"
import { Chart, registerables } from "chart.js"

Chart.register(...registerables)

interface BurnSeverityChartProps {
  data: {
    low: number
    moderate: number
    high: number
    veryHigh: number
    extreme: number
  }
}

export function BurnSeverityChart({ data }: BurnSeverityChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)

  useEffect(() => {
    if (!chartRef.current) return

    // Destroy previous chart instance if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    const ctx = chartRef.current.getContext("2d")
    if (!ctx) return

    const labels = ["Low", "Moderate", "High", "Very High", "Extreme"]
    const values = [data.low, data.moderate, data.high, data.veryHigh, data.extreme]
    const colors = ["#69B34C", "#FAB733", "#FF8E15", "#FF4E11", "#FF0D0D"]

    chartInstance.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Area (km²)",
            data: values,
            backgroundColor: colors,
            borderColor: colors,
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const value = context.raw as number
                const total = values.reduce((a, b) => a + b, 0)
                const percentage = ((value / total) * 100).toFixed(1)
                return `${value.toFixed(2)} km² (${percentage}%)`
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Area (km²)",
            },
          },
        },
      },
    })

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
    }
  }, [data])

  return (
    <div className="w-full h-64">
      <canvas ref={chartRef} />
    </div>
  )
}
