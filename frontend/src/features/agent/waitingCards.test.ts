import { describe, expect, it } from 'vitest'
import type { WaitingCard } from '../../types/agent'
import { askedAgo, groupCards, resumesPlan, tapNote } from './waitingCards'

function card(over: Partial<WaitingCard>): WaitingCard {
  return { id: 'confirm:a', kind: 'app_confirm', question: 'Trash “A”?', asked_at: '2026-11-23T10:00:00Z', group: 2, app: 'tasks', options: [], expires_at: null, latest: false, ...over }
}

describe('waiting cards (#251)', () => {
  it('groups one request’s cards, in the order asked', () => {
    const groups = groupCards([card({ id: 'confirm:a' }), card({ id: 'confirm:b' }), card({ id: 'plan:7', kind: 'gate', group: 9 }), card({ id: 'plan:8', kind: 'proposal', group: null })])
    expect(groups.map((group) => group.cards.map((one) => one.id))).toEqual([['confirm:a', 'confirm:b'], ['plan:7'], ['plan:8']])
  })

  it('a pick-one is never part of Approve all: it needs its own option', () => {
    const [group] = groupCards([card({ id: 'confirm:a' }), card({ id: 'plan:4', kind: 'pick', options: ['A', 'B'] }), card({ id: 'confirm:b' })])
    expect(group.approvable.map((one) => one.id)).toEqual(['confirm:a', 'confirm:b'])
  })

  it('says how long a card has waited', () => {
    const asked = Date.parse('2026-11-23T10:00:00Z')
    expect(askedAgo('2026-11-23T10:00:00Z', asked + 20_000)).toBe('just now')
    expect(askedAgo('2026-11-23T10:00:00Z', asked + 5 * 60_000)).toBe('5 min ago')
    expect(askedAgo('2026-11-23T10:00:00Z', asked + 120 * 60_000)).toBe('2 h ago')
    // The live case that started #251: a question asked 2 h 37 min earlier.
    expect(askedAgo('2026-11-23T10:00:00Z', asked + 157 * 60_000)).toBe('2 h 37 min ago')
    expect(askedAgo('not a date', asked)).toBe('')
  })

  it('writes the tap as the thread records it', () => {
    expect(tapNote([card({})])).toBe('Tapped Approve: Trash “A”?')
    expect(tapNote([card({ question: 'Which one?' })], 'B')).toBe('Tapped “B”: Which one?')
    expect(tapNote([1, 2, 3, 4].map((n) => card({ question: `Q${n}?` })))).toBe('Tapped Approve: Q1? Q2? Q3? (and 1 more)')
  })

  it('a gate, a proposal and a pick-one always run a turn when approved', () => {
    expect(resumesPlan([card({})])).toBe(false)
    expect(resumesPlan([card({}), card({ kind: 'gate' })])).toBe(true)
    expect(resumesPlan([card({ kind: 'pick' })])).toBe(true)
  })
})
