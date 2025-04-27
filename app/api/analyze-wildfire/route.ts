import { type NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import fs from "fs"

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { latitude, longitude, preFireDate, postFireDate, bufferKm = 5, showHotspot = false } = body

    // Validate input
    if (!latitude || !longitude || !preFireDate || !postFireDate) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 })
    }

    console.log("API Request received:", { latitude, longitude, preFireDate, postFireDate, bufferKm, showHotspot })

    // Path to the Python script and results
    const scriptsDir = path.join(process.cwd(), "scripts")
    const outputDir = path.join(scriptsDir, "output")
    const scriptPath = path.join(scriptsDir, "args.py")
    const resultsPath = path.join(outputDir, "results.json")

    // Create output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true })
    }

    // Create images directory if it doesn't exist
    const imagesDir = path.join(outputDir, "images")
    if (!fs.existsSync(imagesDir)) {
      fs.mkdirSync(imagesDir, { recursive: true })
    }

    // Check if the script exists
    if (!fs.existsSync(scriptPath)) {
      return NextResponse.json(
        {
          error: `Python script not found at ${scriptPath}. Please make sure args.py is in the scripts directory.`,
        },
        { status: 500 },
      )
    }

    // Check if Python is installed
    try {
      const pythonVersionCheck = await execAsync("python --version")
      console.log("Python version:", pythonVersionCheck.stdout)
    } catch (pythonError) {
      console.error("Python check error:", pythonError)
      return NextResponse.json(
        {
          error: "Python is not installed or not in PATH. Please install Python and make sure it's in your PATH.",
        },
        { status: 500 },
      )
    }

    // Build the command with proper quoting for paths to handle spaces
    const command = `python "${scriptPath}" --latitude ${latitude} --longitude ${longitude} --pre-fire-date ${preFireDate} --post-fire-date ${postFireDate} --buffer ${bufferKm} --output-dir "${outputDir}"${showHotspot ? ' --show-hotspot' : ''}`

    console.log(`Executing command: ${command}`)

    try {
      // Execute the command with a timeout
      const { stdout, stderr } = await execAsync(command, {
        timeout: 300000, // 5 minute timeout
        maxBuffer: 10 * 1024 * 1024, // Increase buffer size to 10MB
      })

      console.log("Python script stdout:", stdout)
      if (stderr) {
        console.warn("Python script stderr:", stderr)
      }

      // Check if results.json exists
      if (!fs.existsSync(resultsPath)) {
        // Check if error.json exists
        const errorPath = path.join(outputDir, "error.json")
        if (fs.existsSync(errorPath)) {
          try {
            const errorData = fs.readFileSync(errorPath, "utf-8")
            const errorJson = JSON.parse(errorData)
            return NextResponse.json(
              {
                error: `Python script error: ${errorJson.error || "Unknown error"}`,
                details: errorJson,
                stdout,
                stderr,
                command,
              },
              { status: 500 },
            )
          } catch (parseError) {
            console.error("Error parsing error.json:", parseError)
            return NextResponse.json(
              {
                error: "Failed to parse error information from Python script",
                details: String(parseError),
                stdout,
                stderr,
                command,
              },
              { status: 500 },
            )
          }
        }

        return NextResponse.json(
          {
            error: "Analysis failed to produce results. The Python script did not generate a results.json file.",
            stdout,
            stderr,
            command,
          },
          { status: 500 },
        )
      }

      // Read and parse the results file
      let results
      try {
        const resultsData = fs.readFileSync(resultsPath, "utf-8")
        results = JSON.parse(resultsData)
      } catch (parseError) {
        console.error("Error parsing results.json:", parseError)
        return NextResponse.json(
          {
            error: "Failed to parse results from Python script",
            details: String(parseError),
            stdout,
            stderr,
            command,
          },
          { status: 500 },
        )
      }

      // Copy images to public directory
      if (results.images) {
        const publicDir = path.join(process.cwd(), "public", "analysis_images")

        // Create the directory if it doesn't exist
        if (!fs.existsSync(publicDir)) {
          fs.mkdirSync(publicDir, { recursive: true })
        }

        // Create a unique subdirectory for this analysis
        const timestamp = new Date().getTime()
        const analysisDir = path.join(publicDir, `analysis_${timestamp}`)
        fs.mkdirSync(analysisDir, { recursive: true })

        Object.keys(results.images).forEach((key) => {
          const imagePath = path.join(outputDir, results.images[key])

          // Check if the image exists
          if (fs.existsSync(imagePath)) {
            // Create a new path in the public directory
            const filename = path.basename(imagePath)
            const newPath = path.join(analysisDir, filename)

            // Copy the image to the public directory
            try {
              fs.copyFileSync(imagePath, newPath)

              // Update the path to be relative for browser access
              results.images[key] = `/analysis_images/analysis_${timestamp}/${filename}`
            } catch (copyError) {
              console.error(`Error copying image ${imagePath}:`, copyError)
              results.images[key] = `/placeholder.svg?height=400&width=600&text=${key}`
            }
          } else {
            console.warn(`Image not found: ${imagePath}`)
            results.images[key] = `/placeholder.svg?height=400&width=600&text=${key}`
          }
        })
      }

      return NextResponse.json(results)
    } catch (execError: any) {
      console.error("Full execution error:", execError)
      
      const errorResponse = {
        error: "Error executing Python script",
        details: execError.message || String(execError),
        command,
        stdout: execError.stdout || "",
        stderr: execError.stderr || "",
      }

      // Check if error.json exists even in case of execution error
      const errorPath = path.join(outputDir, "error.json")
      if (fs.existsSync(errorPath)) {
        try {
          const errorData = fs.readFileSync(errorPath, "utf-8")
          const errorJson = JSON.parse(errorData)
          errorResponse.details = JSON.stringify(errorJson, null, 2)
        } catch (parseError) {
          console.error("Error parsing error.json:", parseError)
        }
      }

      return NextResponse.json(errorResponse, { status: 500 })
    }
  } catch (error) {
    console.error("Error processing request:", error)
    return NextResponse.json(
      {
        error: "An unexpected error occurred",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    )
  }
}
