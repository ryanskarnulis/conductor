import { describe, expect, it } from 'vitest'
import type { AgentMessage } from '../../types/agent'
import { filingNote, isSortTurn } from './sortTurn'

function assistant(toolCalls: AgentMessage['tool_calls']): AgentMessage {
  return {
    id: 2,
    conversation_id: 1,
    role: 'assistant',
    content: 'yo, Zeds Dead has 5 waiting. where they going?',
    tool_calls: toolCalls,
    stop_reason: 'completed',
    created_at: '2026-08-24T10:00:00Z',
  }
}

describe('isSortTurn', () => {
  it('opens on a turn where music actually ran the sorting tool', () => {
    expect(
      isSortTurn(
        assistant([
          { tool: 'ask_music', arguments: {}, result: 'five waiting', error: null, app_tools: ['sort_music'] },
        ]),
      ),
    ).toBe(true)
  })

  it('stays shut when music was asked something else', () => {
    expect(
      isSortTurn(
        assistant([
          { tool: 'ask_music', arguments: {}, result: 'downloaded it', error: null, app_tools: ['download_song'] },
        ]),
      ),
    ).toBe(false)
  })

  it('is not fooled by a reply that merely talks about sorting', () => {
    // The whole reason this reads the trajectory: a 12B paraphrases its own
    // answer, and a panel that greps the text breaks the first time it does.
    expect(
      isSortTurn(
        assistant([
          {
            tool: 'ask_music',
            arguments: {},
            result: 'I could sort_music for you if you want',
            error: null,
            app_tools: null,
          },
        ]),
      ),
    ).toBe(false)
  })

  it('stays shut when the delegate call failed', () => {
    expect(
      isSortTurn(
        assistant([
          { tool: 'ask_music', arguments: {}, result: null, error: 'music is down', app_tools: ['sort_music'] },
        ]),
      ),
    ).toBe(false)
  })

  it('ignores a turn with no tools and the user’s own turns', () => {
    expect(isSortTurn(assistant(null))).toBe(false)
    expect(isSortTurn(undefined)).toBe(false)
    expect(
      isSortTurn({ ...assistant(null), role: 'user' }),
    ).toBe(false)
  })
})

describe('filingNote', () => {
  it('records a whole group as what the person did', () => {
    expect(
      filingNote({ artist: 'Zeds Dead', genre: 'Dubstep', tracks: [], filed: 5, notFound: [] }),
    ).toBe('Zeds Dead (5 songs) → Dubstep')
  })

  it('names the songs when only some were filed', () => {
    expect(
      filingNote({
        artist: 'Zeds Dead',
        genre: 'Dubstep',
        tracks: ['Collapse', 'Rumble'],
        filed: 2,
        notFound: [],
      }),
    ).toBe('Zeds Dead: Collapse, Rumble → Dubstep')
  })

  it('says what it could not find, rather than quietly filing fewer', () => {
    expect(
      filingNote({
        artist: 'Zeds Dead',
        genre: 'Dubstep',
        tracks: ['Collapse'],
        filed: 1,
        notFound: ['Blackout'],
      }),
    ).toContain("couldn't find Blackout")
  })
})
