import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { WaitingCard } from '../../types/agent'
import { WaitingOnYou } from './WaitingOnYou'

function card(over: Partial<WaitingCard>): WaitingCard {
  return {
    id: 'confirm:a',
    kind: 'app_confirm',
    question: 'Trash “A”?',
    asked_at: new Date(Date.now() - 157 * 60_000 - 1000).toISOString(),
    group: 2,
    app: 'tasks',
    options: [],
    expires_at: null,
    latest: false,
    ...over,
  }
}

describe('WaitingOnYou (#251)', () => {
  it('folds to its head when nothing was just asked, and says how old the oldest card is', () => {
    render(<WaitingOnYou cards={[card({}), card({ id: 'confirm:b', question: 'Trash “B”?' })]} results={[]} busy={false} onAnswer={vi.fn()} />)
    const head = screen.getByRole('button', { name: /Waiting on you/ })
    expect(head).toHaveAttribute('aria-expanded', 'false')
    expect(head).toHaveTextContent('2')
    expect(head).toHaveTextContent('oldest 2 h 37 min ago')
    expect(screen.queryByRole('button', { name: 'Approve' })).toBeNull()
    fireEvent.click(head)
    expect(head).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getAllByRole('button', { name: 'Approve' })).toHaveLength(2)
    expect(screen.getByText('Only a tap approves. Saying “yes” doesn’t.')).toBeInTheDocument()
  })

  it('opens on its own when the turn right before asked, with one Approve all for the group', () => {
    const onAnswer = vi.fn()
    const asked = [card({ latest: true }), card({ id: 'confirm:b', question: 'Trash “B”?', latest: true })]
    render(<WaitingOnYou cards={asked} results={[]} busy={false} onAnswer={onAnswer} />)
    expect(screen.getByRole('button', { name: /Waiting on you/ })).toHaveAttribute('aria-expanded', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Approve all 2' }))
    expect(onAnswer).toHaveBeenCalledWith(asked, 'yes')
    const second = screen.getByRole('group', { name: 'Waiting: Trash “B”?' })
    fireEvent.click(within(second).getByRole('button', { name: 'Not now' }))
    expect(onAnswer).toHaveBeenLastCalledWith([asked[1]], 'no')
  })

  it('a pick-one answers by the place of the option tapped and is left out of Approve all', () => {
    const onAnswer = vi.fn()
    const pick = card({ id: 'plan:4', kind: 'pick', app: null, question: 'Which one?', options: ['A', 'B'], latest: true })
    render(<WaitingOnYou cards={[card({ latest: true }), pick]} results={[]} busy={false} onAnswer={onAnswer} />)
    expect(screen.queryByRole('button', { name: /^Approve all/ })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'B' }))
    expect(onAnswer).toHaveBeenCalledWith([pick], 'yes', 1)
  })

  it('disables every button while a tap or a turn is in flight', () => {
    render(<WaitingOnYou cards={[card({ latest: true })]} results={[]} busy onAnswer={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Approve' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Not now' })).toBeDisabled()
  })

  it('says why a card didn’t go through, also once no card is left', () => {
    const detail = 'tasks did nothing: task “A” was finished since you were asked, so nothing was trashed.'
    render(<WaitingOnYou cards={[]} results={[{ id: 'confirm:a', answer: 'yes', outcome: 'failed', detail }]} busy={false} onAnswer={vi.fn()} />)
    expect(screen.getByRole('status')).toHaveTextContent(detail)
  })

  it('renders nothing when no card waits and nothing failed', () => {
    const { container } = render(<WaitingOnYou cards={[]} results={[{ id: 'confirm:a', answer: 'yes', outcome: 'done', detail: null }]} busy={false} onAnswer={vi.fn()} />)
    expect(container).toBeEmptyDOMElement()
  })
})
