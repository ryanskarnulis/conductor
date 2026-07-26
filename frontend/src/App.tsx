import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { GatewayLink } from './components/GatewayLink'
import { AgentPage } from './features/agent/AgentPage'

// Conductor is a single-feature app: the chat panel IS the app, so both
// routes render it — `/` with no thread open, `/c/:conversationId` with one.
function App() {
  return (
    <BrowserRouter>
      <GatewayLink />
      <Routes>
        <Route path="/" element={<AgentPage />} />
        <Route path="/c/:conversationId" element={<AgentPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
