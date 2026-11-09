import { GatewayLink } from './components/GatewayLink'
import { AgentPage } from './features/agent/AgentPage'

// Conductor is a single-feature app: the timeline IS the app (#82). There is
// one conversation, so no routes: any path (an old `/c/:id` link included)
// opens the timeline, and the entry carries no router.
function App() {
  return (
    <>
      <GatewayLink />
      <AgentPage />
    </>
  )
}

export default App
