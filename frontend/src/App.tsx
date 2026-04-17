import { useState } from 'react'
import { sendMessage, type Engine } from './api'
import { ChatWindow, type ChatMessage } from './components/ChatWindow'
import { EngineToggle } from './components/EngineToggle'
import { MessageInput } from './components/MessageInput'

const ERROR_REPLY = 'Error: could not reach backend.'

/**
 * Root component for the chat prototype. Owns all application state:
 * message history, selected engine, and in-flight request flag.
 */
function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [engine, setEngine] = useState<Engine>('langchain')
  const [loading, setLoading] = useState(false)

  /**
   * Append the user message, call the backend, then append the reply (or
   * an error bubble if the request fails).
   */
  const handleSend = async (text: string) => {
    const userMessage: ChatMessage = { role: 'user', text }
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    try {
      const reply = await sendMessage(text, engine)
      setMessages((prev) => [...prev, { role: 'assistant', text: reply }])
    } catch (error) {
      console.error(error)
      setMessages((prev) => [...prev, { role: 'assistant', text: ERROR_REPLY }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col bg-slate-100 text-slate-900">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3">
        <h1 className="text-lg font-semibold">RAG Chat</h1>
        <EngineToggle value={engine} onChange={setEngine} />
      </header>
      <ChatWindow messages={messages} />
      <MessageInput disabled={loading} onSend={handleSend} />
    </div>
  )
}

export default App
