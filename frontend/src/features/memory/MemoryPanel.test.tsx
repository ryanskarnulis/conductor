import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { sourceLine } from './factSource'
import MemoryPanel from './MemoryPanel'
import { editFact, forgetFact, listFacts, type MemoryFact } from './memoryApi'

vi.mock('./memoryApi', () => ({
  listFacts: vi.fn(),
  editFact: vi.fn(),
  forgetFact: vi.fn(),
}))

const mockList = vi.mocked(listFacts)
const mockEdit = vi.mocked(editFact)
const mockForget = vi.mocked(forgetFact)

function fact(overrides: Partial<MemoryFact> = {}): MemoryFact {
  return {
    id: 1,
    text: 'I sort music on Sundays',
    created_at: '2026-11-03T18:00:00Z',
    updated_at: '2026-11-03T18:00:00Z',
    source: {
      turn_id: 7,
      kind: 'message',
      words: 'remember I sort music on Sundays',
      said_at: '2026-11-03T18:00:00Z',
      surface: 'voice',
      thread: { id: 3, title: 'Weekend plans' },
    },
    ...overrides,
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([fact()])
})

function panel(conversationId: number | null = 3) {
  const onChanged = vi.fn()
  const onClose = vi.fn()
  render(<MemoryPanel conversationId={conversationId} onClose={onClose} onChanged={onChanged} />)
  return { onChanged, onClose }
}

describe('sourceLine', () => {
  it('says what the person said, when, where and in which conversation', () => {
    expect(sourceLine(fact())).toBe(
      'you said “remember I sort music on Sundays” · Tue, Nov 3 · by voice · in “Weekend plans”',
    )
  })

  it('says an edit was theirs, and never shows forgotten words', () => {
    const edited = fact({ source: { ...fact().source!, kind: 'note', words: 'Changed a memory', surface: null } })
    expect(sourceLine(edited)).toBe('you changed it in the memory list · Tue, Nov 3 · in “Weekend plans”')
    const forgotten = fact({ source: { ...fact().source!, words: null, surface: null, thread: null } })
    expect(sourceLine(forgotten)).toBe('you said it (those words were forgotten) · Tue, Nov 3')
    expect(sourceLine(fact({ source: null }))).toBe('you changed it in the memory list · Tue, Nov 3')
  })
})

describe('MemoryPanel', () => {
  it('lists each fact quoted, with where it came from', async () => {
    panel()
    expect(await screen.findByText('I sort music on Sundays')).toBeInTheDocument()
    expect(screen.getByText(/you said “remember I sort music on Sundays”/)).toBeInTheDocument()
    expect(screen.getByRole('list', { name: 'What conductor remembers' })).toBeInTheDocument()
  })

  it('saves an edit as the person’s words in the open conversation', async () => {
    const { onChanged } = panel()
    fireEvent.click(await screen.findByRole('button', { name: 'Edit: I sort music on Sundays' }))
    const input = screen.getByRole('textbox', { name: 'Change this memory' })
    mockEdit.mockResolvedValue({
      fact: fact({ text: 'I sort music on Saturdays', source: { ...fact().source!, kind: 'note' } }),
      message: null,
    })
    fireEvent.change(input, { target: { value: 'I sort music on Saturdays' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => expect(mockEdit).toHaveBeenCalledWith(1, 'I sort music on Saturdays', 3))
    expect(await screen.findByText('I sort music on Saturdays')).toBeInTheDocument()
    expect(onChanged).toHaveBeenCalled()
  })

  it('escape leaves a fact as it was', async () => {
    panel()
    fireEvent.click(await screen.findByRole('button', { name: 'Edit: I sort music on Sundays' }))
    fireEvent.keyDown(screen.getByRole('textbox', { name: 'Change this memory' }), { key: 'Escape' })
    expect(mockEdit).not.toHaveBeenCalled()
    expect(screen.getByText('I sort music on Sundays')).toBeInTheDocument()
  })

  it('forgets a fact and says it is gone from recall and the state block', async () => {
    mockForget.mockResolvedValue({ fact: null, message: null })
    const { onChanged } = panel(null)
    fireEvent.click(await screen.findByRole('button', { name: 'Forget: I sort music on Sundays' }))
    await waitFor(() => expect(mockForget).toHaveBeenCalledWith(1, null))
    expect(await screen.findByText(/Gone from recall and the state block/)).toBeInTheDocument()
    expect(screen.queryByRole('list', { name: 'What conductor remembers' })).not.toBeInTheDocument()
    expect(screen.getByText(/Nothing remembered/)).toBeInTheDocument()
    expect(onChanged).toHaveBeenCalled()
  })

  it('says when the list can’t be read', async () => {
    mockList.mockRejectedValue(new Error('down'))
    panel()
    expect(await screen.findByRole('alert')).toHaveTextContent('Couldn’t read the memory list.')
  })
})
