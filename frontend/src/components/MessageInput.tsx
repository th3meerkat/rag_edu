import { useState, type KeyboardEvent } from 'react'

type MessageInputProps = {
  disabled: boolean
  onSend: (text: string) => void
}

/**
 * Text input plus send button pinned to the bottom of the chat. Submits on
 * Enter (without Shift) and on click. Send is disabled when the input is
 * empty or a request is in flight.
 */
export function MessageInput({ disabled, onSend }: MessageInputProps) {
  const [text, setText] = useState('')

  const canSend = text.trim().length > 0 && !disabled

  const submit = () => {
    if (!canSend) return
    onSend(text.trim())
    setText('')
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-3">
      <div className="mx-auto flex max-w-3xl items-center gap-2">
        <input
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message"
          className="flex-1 rounded-full border border-slate-300 bg-slate-50 px-4 py-2 text-sm outline-none focus:border-slate-500"
          disabled={disabled}
        />
        <button
          type="button"
          onClick={submit}
          disabled={!canSend}
          className="rounded-full bg-green-500 px-5 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-green-600 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Send
        </button>
      </div>
    </div>
  )
}
