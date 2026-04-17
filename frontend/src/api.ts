/**
 * Client for the chat backend.
 *
 * The backend exposes a single POST /chat endpoint that takes a user
 * message and an engine identifier and returns an assistant reply.
 */

export type Engine = 'langchain' | 'llamaindex'

const CHAT_ENDPOINT = 'http://localhost:8000/chat'

/**
 * Send a chat message to the backend and return the assistant reply.
 *
 * @param message - The user message to send.
 * @param engine - Which backend engine should handle the message.
 * @returns The reply string from the backend.
 * @throws If the network request fails or the response is not ok.
 */
export async function sendMessage(
  message: string,
  engine: Engine,
): Promise<string> {
  const response = await fetch(CHAT_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, engine }),
  })

  if (!response.ok) {
    throw new Error(`Backend responded with status ${response.status}`)
  }

  const data = (await response.json()) as { reply: string }
  return data.reply
}
