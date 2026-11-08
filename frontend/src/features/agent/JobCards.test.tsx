import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { JobEvent } from '../../types/agent'
import { JobCards, JobNotes } from './JobCards'
import { applyJobEvent, EMPTY_JOBS, jobCards } from './jobStream'
import type { ShellTurn } from './turnView'

const job = (over: Partial<JobEvent> = {}): JobEvent => ({ app: 'music', ref: '118', status: 'working', done: 12, total: 40, title: 'Late Night Drive', untrusted: true, ended: false, ...over })
const cardsOf = (...jobs: JobEvent[]) => jobCards(jobs.reduce((s, j, i) => applyJobEvent(s, 'job', j, i + 1), EMPTY_JOBS), {}, new Set())
const errors = {}

describe('JobCards', () => {
  it('one job is a card: its words, its numbers and a bar, and Stop asks for that job', () => {
    const onCancel = vi.fn()
    render(<JobCards cards={cardsOf(job({ done: 164, total: 1480 }))} announce={null} errors={errors} onCancel={onCancel} />)
    const card = screen.getByRole('group', { name: 'Job: Late Night Drive' })
    expect(within(card).getByText('working · 164 of 1,480')).toBeInTheDocument()
    expect(within(card).getByRole('progressbar', { name: 'Late Night Drive progress' })).toHaveAttribute('value', '164')
    fireEvent.click(within(card).getByRole('button', { name: 'Stop Late Night Drive' }))
    expect(onCancel).toHaveBeenCalledWith(expect.objectContaining({ app: 'music', ref: '118' }))
  })

  it('queued is said as queued, and an outside title is a quoted label, not markup', () => {
    render(<JobCards cards={cardsOf(job({ status: 'submitted', done: 0, title: '<img src=x onerror=alert(1)>' }))} announce={null} errors={errors} onCancel={vi.fn()} />)
    expect(screen.getByText('queued · 0 of 40')).toBeInTheDocument()
    const label = screen.getByText('<img src=x onerror=alert(1)>')
    expect(label.tagName).toBe('Q')
    expect(label).toHaveAttribute('title', 'This name comes from outside the house')
    expect(document.querySelector('img')).toBeNull()
  })

  it('shows Asking…, then Asked to stop, and the failure when the app can’t be reached', () => {
    const [card] = cardsOf(job())
    const { rerender } = render(<JobCards cards={[{ ...card, asking: 'sending' }]} announce={null} errors={errors} onCancel={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Stop Late Night Drive' })).toHaveTextContent('Asking…')
    expect(screen.getByRole('button', { name: 'Stop Late Night Drive' })).toBeDisabled()
    rerender(<JobCards cards={[{ ...card, asking: 'asked', word: 'stopping' }]} announce={null} errors={errors} onCancel={vi.fn()} />)
    expect(screen.getByText('stopping · 12 of 40')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop Late Night Drive' })).toHaveTextContent('Asked to stop')
    rerender(<JobCards cards={[card]} announce={null} errors={{ 'music:118': 'music couldn’t be reached' }} onCancel={vi.fn()} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Couldn’t stop it: music couldn’t be reached')
    expect(screen.getByRole('button', { name: 'Stop Late Night Drive' })).toBeEnabled()
  })

  it('an ended job has no Stop and says its note is on the way', () => {
    render(<JobCards cards={cardsOf(job({ status: 'completed', done: 38, ended: true, note_message_id: 9 }))} announce={null} errors={errors} onCancel={vi.fn()} />)
    expect(screen.getByText('finished · 38 of 40')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Stop/ })).toBeNull()
    expect(screen.getByText('Its note is on the way.')).toBeInTheDocument()
  })

  it('several jobs are one calm stack: three rows, the rest folded, each with its own Stop', () => {
    const onCancel = vi.fn()
    const jobs = ['1', '2', '3', '4', '5'].map((ref) => job({ ref, title: `Mix ${ref}`, done: Number(ref) }))
    render(<JobCards cards={cardsOf(...jobs)} announce={null} errors={errors} onCancel={onCancel} />)
    const stack = screen.getByRole('region', { name: 'Background jobs' })
    expect(within(stack).getByText('5 jobs in the background')).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: /Job:/ })).toBeNull()
    expect(within(stack).getByText('2 more')).toBeInTheDocument()
    fireEvent.click(within(stack).getByRole('button', { name: 'Stop Mix 2' }))
    expect(onCancel).toHaveBeenCalledWith(expect.objectContaining({ ref: '2' }))
  })

  it('announces a change politely, once, in words', () => {
    render(<JobCards cards={cardsOf(job())} announce="music: “Late Night Drive”, working, 12 of 40." errors={errors} onCancel={vi.fn()} />)
    expect(screen.getByRole('status')).toHaveTextContent('music: “Late Night Drive”, working, 12 of 40.')
    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite')
  })
})

describe('JobNotes', () => {
  const note = (key: string, reply: string): ShellTurn => ({ key, said: '', note: false, ack: null, resumedNote: null, steps: [], receipts: [], pause: null, announce: null, reply, stopReason: null, live: false, jobNote: true, at: '2026-11-06T18:12:00Z' })

  it('one note reads in full, naming what failed', () => {
    render(<JobNotes turns={[note('a', 'Your playlist is in: 38 of 40. “Lost Signal” and “Paper Moons” didn’t match.')]} />)
    expect(screen.getByRole('complementary', { name: 'Job note' })).toHaveTextContent('“Lost Signal” and “Paper Moons” didn’t match.')
  })

  it('several notes are one block, each a line that opens to its note', () => {
    render(<JobNotes turns={[note('a', '**Playlist** is in: 38 of 40.\nTwo missed.'), note('b', 'Library sorted: 1,480 songs.'), note('c', 'Road trip failed: 3 of 12.')]} />)
    const block = screen.getByRole('complementary', { name: 'Job notes' })
    expect(within(block).getByText('3 jobs finished')).toBeInTheDocument()
    expect(within(block).getByText('Playlist is in: 38 of 40.')).toBeInTheDocument()
    expect(block.querySelectorAll('details')).toHaveLength(3)
  })
})
