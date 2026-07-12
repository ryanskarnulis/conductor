import { describe, expect, it } from 'vitest'
import type { TurnActivity } from '../../types/agent'
import { delegateAckText } from './voiceAck'

function beat(overrides: Partial<TurnActivity>): TurnActivity {
  return {
    active: true,
    kind: 'tool',
    tool: 'ask_chess',
    iteration: 1,
    elapsed_seconds: 2.5,
    ...overrides,
  }
}

describe('delegateAckText', () => {
  it('announces the first beat of a delegate call', () => {
    const acked = new Set<string>()
    expect(delegateAckText(beat({}), acked)).toBe('Asking chess…')
    expect(acked.has('chess')).toBe(true)
  })

  it('announces each app at most once per turn', () => {
    const acked = new Set<string>()
    expect(delegateAckText(beat({}), acked)).toBe('Asking chess…')
    expect(delegateAckText(beat({}), acked)).toBeNull()
    expect(delegateAckText(beat({ tool: 'ask_tasks' }), acked)).toBe('Asking tasks…')
    expect(delegateAckText(beat({ tool: 'ask_tasks' }), acked)).toBeNull()
  })

  it('ignores model beats, non-delegate tools, idle polls, and null', () => {
    const acked = new Set<string>()
    expect(delegateAckText(null, acked)).toBeNull()
    expect(delegateAckText(beat({ active: false }), acked)).toBeNull()
    expect(delegateAckText(beat({ kind: 'model', tool: null }), acked)).toBeNull()
    expect(delegateAckText(beat({ tool: 'list_agents' }), acked)).toBeNull()
    expect(acked.size).toBe(0)
  })
})
