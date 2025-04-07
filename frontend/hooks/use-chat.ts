"use client"

import { useState, useCallback } from "react"
import { nanoid } from "nanoid"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  isLoading?: boolean
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }, [])

  const handleSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault()
      
      if (!input.trim() || isLoading) return

      // Add user message to the chat
      const userMessage: Message = {
        id: nanoid(),
        role: "user",
        content: input,
      }
      
      setMessages((prev) => [...prev, userMessage])
      setInput("")
      setIsLoading(true)

      try {
        const response = await fetch("http://api.rycerz.es/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: "llama-3.3-70b-versatile",
            messages: [{ role: "user", content: input }],
            temperature: 0.2,
          }),
        })

        if (!response.ok) {
          throw new Error(`Error: ${response.status}`)
        }

        const data = await response.json()
        
        // Extract the assistant's response from the API response
        const assistantContent = data.choices?.[0]?.message?.content || "Sorry, I couldn't process your request."
        
        // Add assistant message to the chat
        const assistantMessage: Message = {
          id: nanoid(),
          role: "assistant",
          content: assistantContent,
        }

        setMessages((prev) => [...prev, assistantMessage])
      } catch (error) {
        console.error("Error fetching response:", error)
        
        // Add error message
        const errorMessage: Message = {
          id: nanoid(),
          role: "assistant",
          content: "Sorry, there was an error processing your request. Please try again.",
        }
        
        setMessages((prev) => [...prev, errorMessage])
      } finally {
        setIsLoading(false)
      }
    },
    [input, isLoading]
  )

  return {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
  }
}

