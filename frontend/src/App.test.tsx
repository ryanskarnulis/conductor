import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { getConversation, getTimeline } from './api/agent'

vi.mock('./api/agent', () => ({
  // Side threads (#139): none open.
  listThreads: vi.fn(() => Promise.resolve([])),
  getConversation: vi.fn(),
  getTimeline: vi.fn(),
  getTurnActivity: vi.fn(),
  postMessage: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

const empty = {
  id: 1,
  title: 'Timeline',
  created_at: '2026-11-09T10:00:00Z',
  updated_at: '2026-11-09T10:00:00Z',
  messages: [],
}

beforeEach(() => {
  vi.mocked(getTimeline).mockResolvedValue({ ...empty, quiet_gap_seconds: 10_800, incognito_notice: 'Incognito.' })
  vi.mocked(getConversation).mockResolvedValue(empty)
})

describe('App', () => {
  it('opens straight onto the one timeline (#82): no conversation list, no new chat', async () => {
    render(<App />)
    expect(await screen.findByText(/What should the house do\?/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Start a conversation|New conversation/ })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Search the timeline' })).toBeInTheDocument()
  })
})
