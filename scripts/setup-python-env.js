const { execSync } = require("child_process")
const fs = require("fs")
const path = require("path")

// Create scripts directory if it doesn't exist
const scriptsDir = path.join(process.cwd(), "scripts")
if (!fs.existsSync(scriptsDir)) {
  fs.mkdirSync(scriptsDir, { recursive: true })
}

console.log("=== Python Environment Setup ===")

// Check if Python is installed
try {
  console.log("Checking Python installation...")
  const pythonVersion = execSync("python --version", { encoding: "utf8" })
  console.log(`✅ ${pythonVersion.trim()}`)
} catch (error) {
  console.error("❌ Python is not installed or not in PATH")
  console.log("Please install Python from https://www.python.org/downloads/")
  console.log("Make sure to check 'Add Python to PATH' during installation")
  process.exit(1)
}

// Install required packages
try {
  console.log("\nInstalling required Python packages...")
  execSync("pip install earthengine-api numpy matplotlib pillow requests", { stdio: "inherit" })
  console.log("✅ Packages installed successfully")
} catch (error) {
  console.error("❌ Error installing packages:", error.message)
  console.log("Try running the command manually:")
  console.log("pip install earthengine-api numpy matplotlib pillow requests")
  process.exit(1)
}

// Authenticate with Earth Engine
try {
  console.log("\nAuthenticating with Google Earth Engine...")
  console.log("This will open a browser window. Please follow the instructions to authenticate.")
  execSync("earthengine authenticate", { stdio: "inherit" })
  console.log("✅ Earth Engine authentication completed")
} catch (error) {
  console.error("❌ Error authenticating with Earth Engine:", error.message)
  console.log("Try running the command manually:")
  console.log("earthengine authenticate")
  process.exit(1)
}

console.log("\n=== Setup completed successfully ===")
console.log("You can now run the wildfire analysis application")
