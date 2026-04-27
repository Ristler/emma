
import { useRef, useState } from 'react'
import './App.css'
import MessageList from './components/MessageList'
import Composer from './components/Composer'
import ModelParamsPanel from './components/ModelParamsPanel'


function App() {
  const [messages, setMessages] = useState([])
  const [prompt, setPrompt] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)
  const [paramsOpen, setParamsOpen] = useState(false)
  const [modelParams, setModelParams] = useState({ temperature: 0.5, maxTokens: 80 })
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID())
  const [cutOff, setCutOff] = useState(false)
  const abortControllerRef = useRef(null)


  async function sendPrompt(userPrompt) {
    const trimmed = userPrompt.trim()
    if (!trimmed || isSending) return

    if (trimmed === '/reset') {
      setSessionId(crypto.randomUUID())
      setCutOff(false)
      setMessages((prev) => [...prev, { role: 'assistant', content: 'History cleared.', timestamp: new Date().toISOString() }])
      setPrompt('')
      return
    }

    // Abort any previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    setError('')
    setIsSending(true)
    setCutOff(false)

    const userMessage = { role: 'user', content: trimmed, timestamp: new Date().toISOString() }
    setMessages((prev) => [...prev, userMessage])
    setPrompt('')

    // Streaming implementation
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: trimmed,
          temperature: modelParams.temperature,
          num_tokens: modelParams.maxTokens,
          session_id: sessionId,
          stream: true
        }),
        signal: abortController.signal
      })
      if (!res.body) throw new Error('No response body')

      let assistantMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, assistantMessage])

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let done = false
      let tokenCount = 0
      while (!done) {
        const { value, done: doneReading } = await reader.read()
        done = doneReading
        if (value) {
          const chunk = decoder.decode(value)
          assistantMessage = {
            ...assistantMessage,
            content: assistantMessage.content + chunk
          }
          tokenCount++
          setMessages((prev) => {
            // Replace the last assistant message with the updated one
            const lastUserIdx = prev.map(m => m.role).lastIndexOf('user')
            const before = prev.slice(0, lastUserIdx + 1)
            return [...before, assistantMessage]
          })
        }
      }
      // If we streamed exactly maxTokens, likely cut off
      if (tokenCount >= modelParams.maxTokens) {
        setCutOff(true)
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        // Request was aborted, do not show error
        return
      }
      setError(e instanceof Error ? e.message : String(e))
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry — something went wrong sending that message.',
          timestamp: new Date().toISOString(),
        },
      ])
    } finally {
      setIsSending(false)
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    void sendPrompt(prompt)
  }

  function onComposerKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void sendPrompt(prompt)
    }
  }

  return (
    <div className="app">
      <main className="chat" aria-label="Chat">
        <div className="chatbox-area">
          <MessageList messages={messages} isSending={isSending} bottomRef={bottomRef} />
          {cutOff && (
            <div style={{ color: 'orange', margin: '8px 0', textAlign: 'center' }}>
              Reply cut off — try increasing max tokens for longer answers.
            </div>
          )}
        </div>
        <div className="composer-area">
          <Composer
            prompt={prompt}
            setPrompt={setPrompt}
            onSubmit={onSubmit}
            onComposerKeyDown={onComposerKeyDown}
            isSending={isSending}
            error={error}
            onOpenParams={() => setParamsOpen(true)}
          />
        </div>
      </main>
      <ModelParamsPanel
        open={paramsOpen}
        onClose={() => setParamsOpen(false)}
        params={modelParams}
        setParams={setModelParams}
      />
    </div>
  )
}

export default App
