const { execSync } = require("child_process")
const fs = require("fs")
const path = require("path")

console.log("=== Python Environment Check ===")

// Check if Python is installed
try {
  console.log("Checking Python installation...")
  const pythonVersion = execSync("python --version", { encoding: "utf8" })
  console.log(`✅ ${pythonVersion.trim()}`)
} catch (error) {
  console.error("❌ Python is not installed or not in PATH:", error.message)
  console.log("Please install Python and make sure it's in your PATH")
  process.exit(1)
}

// Check if pip is available
try {
  console.log("\nChecking pip installation...")
  const pipVersion = execSync("pip --version", { encoding: "utf8" })
  console.log(`✅ ${pipVersion.trim()}`)
} catch (error) {
  console.error("❌ pip is not installed or not in PATH:", error.message)
  console.log("Please install pip (Python package manager)")
  process.exit(1)
}

// Check if the args.py script exists
const scriptsDir = path.join(process.cwd(), "scripts")
const scriptPath = path.join(scriptsDir, "args.py")

if (!fs.existsSync(scriptPath)) {
  console.error(`\n❌ Python script not found at ${scriptPath}`)
  console.log("Please make sure the args.py script is in the scripts directory")
  console.log("You can run 'npm run copy-python-script' to create it")
  process.exit(1)
} else {
  console.log(`\n✅ Python script found at ${scriptPath}`)
}

// Check required packages
console.log("\nChecking required Python packages...")
const requiredPackages = ["earthengine-api", "numpy", "matplotlib", "pillow", "requests"]

for (const pkg of requiredPackages) {
  try {
    execSync(`pip show ${pkg}`, { stdio: "pipe" })
    console.log(`✅ ${pkg} is installed`)
  } catch (error) {
    console.log(`❌ ${pkg} is not installed`)
    console.log(`   You can install it with: pip install ${pkg}`)
  }
}

// Try to run the script with --help to check if it works
try {
  console.log("\nTesting Python script...")
  const helpOutput = execSync(`python "${scriptPath}" --help`, { encoding: "utf8" })
  console.log("✅ Python script test successful!")
} catch (error) {
  console.error("❌ Error testing Python script:", error.message)
  console.log("Please check your Python environment and script dependencies")
  process.exit(1)
}

// Check Earth Engine authentication
console.log("\nChecking Earth Engine authentication...")
try {
  // This command will fail if not authenticated
  execSync("earthengine asset list", { stdio: "pipe" })
  console.log("✅ Earth Engine authentication is valid")
} catch (error) {
  console.log("❌ Earth Engine authentication is missing or invalid")
  console.log("   You need to authenticate with: earthengine authenticate")
}

// Check if the output directory exists and is writable
const outputDir = path.join(process.cwd(), "public", "analysis_results")
console.log("\nChecking output directory...")
try {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true })
    console.log(`✅ Created output directory at ${outputDir}`)
  } else {
    console.log(`✅ Output directory exists at ${outputDir}`)
  }

  // Test write permissions by creating a test file
  const testFile = path.join(outputDir, "test.txt")
  fs.writeFileSync(testFile, "Test write permissions", "utf8")
  fs.unlinkSync(testFile) // Delete the test file
  console.log("✅ Output directory is writable")
} catch (error) {
  console.error("❌ Error with output directory:", error.message)
  console.log("   Make sure the application has write permissions to the public/analysis_results directory")
}

console.log("\n=== Python setup check completed ===")
console.log("If you're still having issues, try running:")
console.log("1. npm run setup-python")
console.log("2. earthengine authenticate")
console.log("3. npm run check-python")
