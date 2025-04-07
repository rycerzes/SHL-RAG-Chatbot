"use client"

import { cn } from "@/lib/utils"
import { useEffect, useState } from "react"

export function TypingIndicator() {
  const [dots, setDots] = useState(1)

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev % 3) + 1)
    }, 500)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex items-center h-5">
      <div className="flex space-x-1">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className={cn("h-2 w-2 rounded-full bg-zinc-400 transition-opacity duration-300", i > dots && "opacity-30")}
          />
        ))}
      </div>
    </div>
  )
}

