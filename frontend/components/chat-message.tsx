"use client"

import { cn } from "@/lib/utils"
import { Avatar } from "@/components/ui/avatar"
import { User, Bot } from "lucide-react"
import { RecommendationTable } from "@/components/recommendation-table"
import { extractRecommendations } from "@/lib/extract-recommendations"
import { TypingIndicator } from "@/components/typing-indicator"
import ReactMarkdown from "react-markdown"

interface ChatMessageProps {
  message: {
    id: string
    role: "user" | "assistant"
    content: string
    isLoading?: boolean
  }
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"
  const recommendations = !isUser ? extractRecommendations(message.content) : null

  return (
    <div className={cn("flex w-full items-start gap-x-4", isUser && "justify-end")}>
      {!isUser && (
        <Avatar className="h-8 w-8 bg-primary">
          <Bot className="h-4 w-4" />
        </Avatar>
      )}
      <div className={cn("flex flex-col max-w-[80%] md:max-w-[70%]", isUser && "items-end")}>
        <div
          className={cn(
            "rounded-lg px-4 py-2 text-sm",
            isUser ? "bg-primary text-primary-foreground" : "bg-zinc-800 text-zinc-100",
          )}
        >
          {message.isLoading ? (
            <TypingIndicator />
          ) : (
            <div className="whitespace-pre-wrap break-words prose prose-invert prose-sm max-w-none">
              <ReactMarkdown>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {recommendations && recommendations.length > 0 && (
          <div className="mt-3 w-full">
            <RecommendationTable recommendations={recommendations} />
          </div>
        )}
      </div>

      {isUser && (
        <Avatar className="h-8 w-8 bg-zinc-700">
          <User className="h-4 w-4" />
        </Avatar>
      )}
    </div>
  )
}

