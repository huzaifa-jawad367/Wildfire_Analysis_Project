"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { AlertTriangle, Info } from "lucide-react"
import { useState } from "react"

export function PythonSetupGuide() {
  const [showGuide, setShowGuide] = useState(true)

  if (!showGuide) return null

  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-amber-500" />
          Python Setup Required
        </CardTitle>
        <CardDescription>
          The wildfire analysis requires Python and several dependencies to be installed
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert variant="default" className="bg-blue-50 border-blue-200">
          <Info className="h-4 w-4 text-blue-500" />
          <AlertTitle className="text-blue-700">Why Python?</AlertTitle>
          <AlertDescription className="text-blue-600">
            This application uses Python to process satellite imagery from Google Earth Engine and calculate burn
            severity metrics. The Python script handles the complex geospatial analysis that would be difficult to
            perform in JavaScript.
          </AlertDescription>
        </Alert>

        <div className="space-y-2">
          <h3 className="font-medium">Setup Instructions:</h3>
          <ol className="list-decimal pl-5 space-y-2">
            <li>
              <strong>Install Python</strong> - Download and install Python 3.7+ from{" "}
              <a
                href="https://www.python.org/downloads/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                python.org
              </a>
            </li>
            <li>
              <strong>Install Required Packages</strong> - Open a command prompt and run:
              <pre className="bg-gray-100 p-2 rounded mt-1 text-sm overflow-x-auto">
                pip install earthengine-api numpy matplotlib pillow requests
              </pre>
            </li>
            <li>
              <strong>Authenticate with Google Earth Engine</strong> - Run the following command and follow the
              instructions:
              <pre className="bg-gray-100 p-2 rounded mt-1 text-sm overflow-x-auto">earthengine authenticate</pre>
            </li>
            <li>
              <strong>Run the Diagnostic Script</strong> - In your project directory, run:
              <pre className="bg-gray-100 p-2 rounded mt-1 text-sm overflow-x-auto">
                node scripts/python-diagnostic.js
              </pre>
            </li>
          </ol>
        </div>

        <div className="space-y-2">
          <h3 className="font-medium">Common Issues:</h3>
          <ul className="list-disc pl-5 space-y-1">
            <li>
              <strong>Python not in PATH</strong> - Make sure Python is added to your system PATH during installation
            </li>
            <li>
              <strong>Permission issues</strong> - Try running the command prompt as administrator
            </li>
            <li>
              <strong>Earth Engine authentication</strong> - You need a Google account with Earth Engine access
            </li>
            <li>
              <strong>Future dates</strong> - Satellite imagery is not available for future dates
            </li>
          </ul>
        </div>

        <div className="flex justify-between">
          <Button variant="outline" onClick={() => setShowGuide(false)}>
            Dismiss Guide
          </Button>
          <Button
            onClick={() => window.open("https://developers.google.com/earth-engine/guides/python_install", "_blank")}
          >
            Earth Engine Docs
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
