import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { listConversations } from './api/agent'

vi.mock('./api/agent', () => ({
  createConversation: vi.fn(),
  deleteConversation: vi.fn(),
  getConversation: vi.fn(),
  getTurnActivity: vi.fn(),
  listConversations: vi.fn(),
  postMessage: vi.fn(),
}))

beforeEach(() => {
  vi.mocked(listConversations).mockResolvedValue([])
})

describe('App', () => {
  it('renders the chat shell with the empty-thread welcome at /', async () => {
    render(<App />)
    expect(await screen.findByRole('heading', { name: 'Conductor' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Start a conversation/ })).toBeInTheDocument()
    expect(await screen.findByText('No conversations yet.')).toBeInTheDocument()
  })
})
