import { useEffect, useRef } from 'react'

export type ChatMessage = {
  role: 'user' | 'assistant'
  text: string
}

type ChatWindowProps = {
  messages: ChatMessage[]
}

/**
 * Scrollable list of chat bubbles. User messages are right-aligned with a
 * green background, assistant messages are left-aligned with a neutral
 * background. Auto-scrolls to the newest message when the list changes.
 */
export function ChatWindow({ messages }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 px-4 py-4">
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        {messages.map((message, index) => (
          <Bubble key={index} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

type BubbleProps = {
  message: ChatMessage
}

/**
 * Single chat bubble. Styling varies by role.
 */
function Bubble({ message }: BubbleProps) {
  const isUser = message.role === 'user'
  const alignment = isUser ? 'self-end' : 'self-start'
  const colors = isUser
    ? 'bg-green-500 text-white rounded-br-sm'
    : 'bg-white text-slate-900 border border-slate-200 rounded-bl-sm'

  return (
    <div
      className={`max-w-[75%] whitespace-pre-wrap break-words rounded-2xl px-3 py-2 text-sm shadow-sm ${alignment} ${colors}`}
    >
      {message.text}
    </div>
  )
}
