import { FireExtinguisher } from "lucide-react"

export function WildfireHeader() {
  return (
    <header className="bg-black text-white p-4">
      <div className="container mx-auto flex items-center gap-2">
        <FireExtinguisher className="h-6 w-6 text-red-500" />
        <h1 className="text-xl font-bold">Wildfire Analysis Dashboard</h1>
      </div>
    </header>
  )
}
