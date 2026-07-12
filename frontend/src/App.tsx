import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AgentPage } from './features/agent/AgentPage'

// Conductor is a single-feature app: the chat panel IS the app, so both
// routes render it — `/` with no thread open, `/c/:conversationId` with one.
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AgentPage />} />
        <Route path="/c/:conversationId" element={<AgentPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
