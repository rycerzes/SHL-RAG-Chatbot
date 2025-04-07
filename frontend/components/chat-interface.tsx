"use client"

import { useChat } from "@/hooks/use-chat"
import { useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar } from "@/components/ui/avatar"
import { Send } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"
import { useMobile } from "@/hooks/use-mobile"

export function ChatInterface() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat()
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const isMobile = useMobile()

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current
      scrollContainer.scrollTop = scrollContainer.scrollHeight
    }
  }, [messages])

  return (
    <Card className="w-full h-full border-zinc-800 bg-zinc-950 flex flex-col">
      <CardHeader className="p-4 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center space-x-4">
          <Avatar className="h-8 w-8 bg-primary">
            <span className="text-xs font-bold">SHL</span>
          </Avatar>
          <div>
            <h2 className="text-sm font-semibold">SHL Solutions Assistant</h2>
            <p className="text-xs text-muted-foreground">Powered by RAG API</p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0 flex-grow overflow-hidden">
        <ScrollArea className="h-full p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <p className="mb-2">Welcome to SHL Solutions Assistant</p>
              <p className="text-sm">Ask me about SHL test solutions and recommendations</p>
            </div>
          ) : (
            <div className="flex flex-col space-y-4">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isLoading && (
                <ChatMessage
                  message={{
                    id: "loading",
                    role: "assistant",
                    content: "",
                    isLoading: true,
                  }}
                />
              )}
            </div>
          )}
        </ScrollArea>
      </CardContent>
      <CardFooter className="p-4 border-t border-zinc-800 flex-shrink-0">
        <form onSubmit={handleSubmit} className="flex w-full space-x-2">
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask about SHL test solutions..."
            className="flex-1 bg-zinc-900 border-zinc-800 focus-visible:ring-primary"
            disabled={isLoading}
          />
          <Button type="submit" size={isMobile ? "icon" : "default"} disabled={isLoading || !input.trim()}>
            {isMobile ? <Send className="h-4 w-4" /> : "Send"}
          </Button>
        </form>
      </CardFooter>
    </Card>
  )
}

