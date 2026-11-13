import { beforeEach, describe, expect, it, vi } from 'vitest'
import { apiClient } from './client'
import { postMessage } from './agent'

vi.mock('./client', () => ({ apiClient: vi.fn(), apiUrl: (path: string) => path }))

beforeEach(() => {
  vi.mocked(apiClient).mockReset().mockResolvedValue(new Response('{}'))
})

function sentBody(): unknown {
  const init = vi.mocked(apiClient).mock.calls[0][1] as RequestInit
  return JSON.parse(String(init.body))
}

describe('postMessage', () => {
  it('sends the heard report in the API’s snake case (#77)', async () => {
    await postMessage(3, 'no, sunday', {
      surface: 'voice',
      heard: { playedMs: 1000, totalMs: null, heardChars: null, totalChars: 100 },
    })
    expect(sentBody()).toEqual({
      content: 'no, sunday',
      surface: 'voice',
      heard: { played_ms: 1000, total_ms: null, heard_chars: null, total_chars: 100 },
    })
  })

  it('sends no heard field when there is none', async () => {
    await postMessage(3, 'hi', { surface: 'voice' })
    expect(sentBody()).toEqual({ content: 'hi', surface: 'voice' })
  })
})
