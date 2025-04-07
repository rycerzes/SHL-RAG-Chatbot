import { ChatInterface } from "@/components/chat-interface"

export default function Home() {
  return (
    <main className="flex flex-col items-center h-screen overflow-hidden">
      <div className="z-10 w-full max-w-5xl flex flex-col h-full px-4 md:px-24 py-4">
        <h1 className="mb-4 text-2xl font-bold text-center flex-shrink-0">SHL Solutions AI Assistant</h1>
        <div className="flex-grow overflow-hidden">
          <ChatInterface />
        </div>
      </div>
    </main>
  )
}

