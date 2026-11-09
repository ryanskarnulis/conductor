import { describe, expect, it } from 'vitest'
import type { AgentMessage } from '../../types/agent'
import { dayLabel, daySeparators, incognitoLapsed, isIncognitoPlaceholder, surfaceLabel, withLiveWords } from './timeline'

function message(overrides: Partial<AgentMessage>): AgentMessage {
  return { id: 1, conversation_id: 1, role: 'user', content: 'hi', tool_calls: null, stop_reason: null, created_at: '2026-11-09T18:00:00Z', ...overrides }
}

describe('the timeline helpers (#82)', () => {
  it('tags each entry with its surface, typed web turns as web', () => {
    expect(surfaceLabel(null)).toBe('web')
    expect(surfaceLabel(undefined)).toBe('web')
    expect(surfaceLabel('desktop')).toBe('web')
    expect(surfaceLabel('voice')).toBe('voice')
    expect(surfaceLabel('room_speaker')).toBe('room speaker')
  })

  it('names a separator Today, Yesterday or the date', () => {
    const now = new Date(2026, 10, 9, 15, 0)
    expect(dayLabel(new Date(2026, 10, 9, 8, 0).toISOString(), now)).toBe('Today')
    expect(dayLabel(new Date(2026, 10, 8, 23, 0).toISOString(), now)).toBe('Yesterday')
    expect(dayLabel(new Date(2026, 10, 3, 9, 0).toISOString(), now)).toBe('Tue, Nov 3')
    expect(dayLabel(new Date(2025, 10, 3, 9, 0).toISOString(), now)).toContain('2025')
  })

  it('puts a separator where the day changes, and only there', () => {
    const times = [
      new Date(2026, 10, 8, 9).toISOString(),
      new Date(2026, 10, 8, 21).toISOString(),
      null,
      new Date(2026, 10, 9, 7).toISOString(),
    ]
    expect([...daySeparators(times).keys()]).toEqual([0, 3])
  })

  it('lapses incognito after the quiet gap since it was last used', () => {
    const gap = 3 * 60 * 60 * 1000
    expect(incognitoLapsed(null, 10, gap)).toBe(false)
    expect(incognitoLapsed(0, gap - 1, gap)).toBe(false)
    expect(incognitoLapsed(0, gap, gap)).toBe(true)
  })

  it('shows this session’s incognito words, never what was written', () => {
    const stored = [message({ id: 1, content: '(incognito)', incognito: true }), message({ id: 2, content: 'kept' })]
    expect(isIncognitoPlaceholder(stored[0])).toBe(true)
    const shown = withLiveWords(stored, new Map([[1, 'the surprise party']]))
    expect(shown.map((m) => m.content)).toEqual(['the surprise party', 'kept'])
    expect(shown[0].incognito).toBe(true)
    expect(withLiveWords(stored, new Map())).toBe(stored)
  })
})
