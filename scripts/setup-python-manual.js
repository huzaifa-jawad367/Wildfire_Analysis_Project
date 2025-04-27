const { execSync } = require("child_process")
const fs = require("fs")
const path = require("path")

// Create scripts directory if it doesn't exist
const scriptsDir = path.join(process.cwd(), "scripts")
if (!fs.existsSync(scriptsDir)) {
  fs.mkdirSync(scriptsDir, { recursive: true })
}

try {
  // Install required Python packages
  console.log("Installing required Python packages...")
  execSync("pip install earthengine-api numpy matplotlib pillow requests", { stdio: "inherit" })
  console.log("Python packages installed successfully")

  // Authenticate with Earth Engine (this will require user interaction)
  console.log("Authenticating with Google Earth Engine...")
  execSync("earthengine authenticate", { stdio: "inherit" })
  console.log("Earth Engine authentication completed")
} catch (error) {
  console.error("Error setting up Python environment:", error.message)
  console.log("Please set up the Python environment manually:")
  console.log("1. Install required packages: pip install earthengine-api numpy matplotlib pillow requests")
  console.log("2. Authenticate with Earth Engine: earthengine authenticate")
}
