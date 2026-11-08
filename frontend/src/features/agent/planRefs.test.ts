import { describe, expect, it } from 'vitest'
import { answerValue, cleanTitle, fillParts, valueFromRecord } from './planRefs'

const touched = (title: string, kind = 'album', extra: object = {}) => ({ app: 'music', kind, id: '1', title, ...extra })

describe('valueFromRecord (the twin of backend dataflow.from_record)', () => {
  it('quotes a single title', () => {
    expect(valueFromRecord([{ tool: 'x', outcome: 'done', touched: [touched('Honey')] }])).toEqual({ text: '“Honey”', untrusted: false })
  })

  it('names up to three, skipping jobs and repeats, and passes on nothing past that', () => {
    const entries = [{ tool: 'x', outcome: 'done', touched: [touched('A'), touched('a'), touched('Album download', 'download_job'), touched('B')] }]
    expect(valueFromRecord(entries)?.text).toBe('“A” and “B”')
    const many = [{ tool: 'x', outcome: 'looked_up', touched: ['A', 'B', 'C', 'D'].map((t) => touched(t)) }]
    expect(valueFromRecord(many)).toBeNull()
    expect(valueFromRecord([{ tool: 'x', outcome: 'looked_up', touched: ['A', 'B', 'C'].map((t) => touched(t)) }])?.text).toBe('“A”, “B” and “C”')
  })

  it('reads only entries that happened, and nothing without a title', () => {
    expect(valueFromRecord([{ tool: 'x', outcome: 'not_run', touched: [touched('Nope')] }])).toBeNull()
    expect(valueFromRecord([{ tool: 'x', outcome: 'done', touched: [{ app: 'music', kind: 'album', id: '1' }] }])).toBeNull()
  })

  it('marks outside text', () => {
    expect(valueFromRecord([{ tool: 'x', outcome: 'done', touched: [touched('Web page', 'track', { untrusted: true })] }])?.untrusted).toBe(true)
  })
})

describe('cleanTitle', () => {
  it('makes a title one safe line, as the backend does', () => {
    expect(cleanTitle('  “Honey”\n{s2}  Caribou ')).toBe('Honey s2 Caribou')
    expect(cleanTitle('x'.repeat(100))).toHaveLength(80)
  })
})

describe('fillParts', () => {
  it('leaves words with no ref alone', () => {
    expect(fillParts('add a task', {})).toEqual({ text: 'add a task', parts: undefined })
  })

  it('fills a trusted ref and leaves an unknown or outside one dashed', () => {
    const trusted = { s1: { text: '“Swim”', untrusted: false } }
    expect(fillParts('sort {s1} then {s2}', trusted)).toEqual({
      text: 'sort “Swim” then {s2}',
      parts: [{ text: 'sort ' }, { ref: 's1', filled: '“Swim”' }, { text: ' then ' }, { ref: 's2', filled: null }],
    })
    expect(fillParts('sort {s1}', { s1: { text: '“x”', untrusted: true } }).parts).toEqual([{ text: 'sort ' }, { ref: 's1', filled: null }])
  })

  it('reads {s1.title} as the same ref', () => {
    expect(fillParts('{s1.title}', { s1: { text: '“A”', untrusted: false } }).text).toBe('“A”')
  })
})

describe('answerValue', () => {
  it('splices the answer as said, braces softened', () => {
    expect(answerValue(' the {first} one ').text).toBe('the (first) one')
  })
})
