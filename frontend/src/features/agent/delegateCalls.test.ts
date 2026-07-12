import { describe, expect, it } from 'vitest'
import type { ToolCallRecord, TurnActivity } from '../../types/agent'
import { activityLabel, appFromTool, describeToolCall, elapsedSuffix } from './delegateCalls'

function record(overrides: Partial<ToolCallRecord>): ToolCallRecord {
  return { tool: 'ask_chess', arguments: {}, result: 'ok', error: null, ...overrides }
}

function beat(overrides: Partial<TurnActivity>): TurnActivity {
  return {
    active: true,
    kind: 'model',
    tool: null,
    iteration: 1,
    elapsed_seconds: 1,
    ...overrides,
  }
}

describe('appFromTool', () => {
  it('extracts the app behind an ask tool, restoring word breaks', () => {
    expect(appFromTool('ask_chess')).toBe('chess')
    expect(appFromTool('ask_home_media')).toBe('home media')
  })

  it('returns null for non-delegate tools', () => {
    expect(appFromTool('list_agents')).toBeNull()
    expect(appFromTool('ask_')).toBeNull()
  })
})

describe('describeToolCall', () => {
  it('labels delegate calls, the fleet lookup, and unknown tools', () => {
    expect(describeToolCall(record({ tool: 'ask_tasks' }))).toBe('Asked tasks')
    expect(describeToolCall(record({ tool: 'list_agents' }))).toBe('Checked the fleet')
    expect(describeToolCall(record({ tool: 'mystery' }))).toBe('Ran mystery')
  })
})

describe('activityLabel', () => {
  it('reads as thinking before the first beat and on model turns', () => {
    expect(activityLabel(null)).toBe('Thinking…')
    expect(activityLabel(beat({ kind: 'model' }))).toBe('Thinking…')
  })

  it('names the app while a delegate call is in flight', () => {
    expect(activityLabel(beat({ kind: 'tool', tool: 'ask_chess' }))).toBe('Asking chess…')
    expect(activityLabel(beat({ kind: 'tool', tool: 'list_agents' }))).toBe(
      'Checking the fleet…',
    )
  })
})

describe('elapsedSuffix', () => {
  it('stays blank for quick turns, then shows rounded seconds', () => {
    expect(elapsedSuffix(null)).toBe('')
    expect(elapsedSuffix(beat({ elapsed_seconds: 3 }))).toBe('')
    expect(elapsedSuffix(beat({ elapsed_seconds: 14.5 }))).toBe(' · 15s')
  })
})
