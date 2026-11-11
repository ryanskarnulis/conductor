import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import {
  ConfirmationCard,
  FinishedNote,
  JobCard,
  Ledger,
  MemoryFact,
  Narration,
  PlanSteps,
  Presence,
  ProposalCard,
  QuestionCard,
  Receipts,
  SideThread,
  StatusPill,
  ThreadHead,
  ThreadNote,
  ThreadStart,
  ThreadTabs,
  Utterance,
} from './components'
import { STEP_WORD, type StepState } from './status'

describe('design system components', () => {
  it('writes every status as a word, never colour alone', () => {
    const states = Object.keys(STEP_WORD) as StepState[]
    render(
      <div>
        {states.map((state) => (
          <StatusPill key={state} state={state} />
        ))}
      </div>,
    )
    for (const state of states) expect(screen.getByText(STEP_WORD[state])).toHaveAttribute('data-state', state)
  })

  it('adds a failure reason after the word', () => {
    render(<StatusPill state="failed" detail="chess didn’t answer" />)
    expect(screen.getByText('failed · chess didn’t answer')).toBeInTheDocument()
  })

  it('renders Presence as a live status only when asked', () => {
    const { rerender } = render(<Presence state="wait" label="Waiting on you" live />)
    expect(screen.getByRole('status')).toHaveTextContent('Waiting on you')
    rerender(<Presence state="idle" label="Ready" />)
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('lists plan steps with kind, app, text and status', () => {
    render(
      <PlanSteps
        steps={[
          { kind: 'ask', app: 'music', text: 'download “Honey”', state: 'job' },
          { kind: 'ask', app: 'PCC', text: 'add task', state: 'wait' },
        ]}
      />,
    )
    const list = screen.getByRole('list', { name: 'Plan steps' })
    expect(list).toHaveTextContent('askmusicdownload “Honey”running · job')
    expect(list).toHaveTextContent('askPCCadd taskwaiting on you')
  })

  it('shows a ref dashed until it is filled, and says so in words', () => {
    const { rerender } = render(
      <PlanSteps steps={[{ kind: 'ask', app: 'PCC', text: 'sort {s1}', state: 'queued', parts: [{ text: 'sort ' }, { ref: 's1', filled: null }] }]} />,
    )
    const ref = screen.getByText('what step 1 finds').closest('.cd-ref')
    expect(ref).toHaveAttribute('data-filled', 'false')
    expect(screen.getByText('{s1}')).toHaveAttribute('aria-hidden', 'true')
    rerender(<PlanSteps steps={[{ kind: 'ask', app: 'PCC', text: 'sort “Honey”', state: 'queued', parts: [{ text: 'sort ' }, { ref: 's1', filled: '“Honey”' }] }]} />)
    expect(screen.getByText('“Honey”')).toHaveAttribute('data-filled', 'true')
    expect(screen.queryByText('what step 1 finds')).not.toBeInTheDocument()
  })

  it('disables a question’s choices while the turn is busy', () => {
    render(<QuestionCard question="Which?" choices={[{ label: 'A' }]} onChoose={vi.fn()} disabled />)
    expect(screen.getByRole('button', { name: 'A' })).toBeDisabled()
  })

  it('shows Undo only when a receipt has a handle and a handler', () => {
    const onUndo = vi.fn()
    render(
      <Receipts
        receipts={[
          { verb: 'Created', app: 'PCC', entity: '“Sort Honey”', undoHow: 'move it to the trash' },
          { verb: 'Queued', app: 'music', entity: '“Honey”', noUndo: 'downloaded files stay on disk' },
        ]}
        onUndo={onUndo}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    expect(onUndo).toHaveBeenCalledWith(expect.objectContaining({ verb: 'Created' }))
    expect(screen.getAllByRole('button', { name: 'Undo' })).toHaveLength(1)
    expect(screen.getByText(/can’t undo: downloaded files stay on disk/)).toBeInTheDocument()
  })

  it('keeps an app’s entity text as text', () => {
    render(<Receipts receipts={[{ verb: 'Found', app: 'PCC', entity: '<img src=x onerror=alert(1)>' }]} />)
    expect(screen.getByText(/<img src=x onerror=alert\(1\)>/)).toBeInTheDocument()
    expect(document.querySelector('img')).toBeNull()
  })

  it('answers a confirmation by its named action or Not now', () => {
    const onConfirm = vi.fn()
    const onDecline = vi.fn()
    render(<ConfirmationCard app="PCC" question="Trash Kitchen?" why="This writes to PCC." actionLabel="Trash it" onConfirm={onConfirm} onDecline={onDecline} />)
    expect(screen.getByRole('group', { name: 'Confirmation' })).toHaveTextContent('Waiting on you · PCC')
    fireEvent.click(screen.getByRole('button', { name: 'Trash it' }))
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }))
    expect(onConfirm).toHaveBeenCalledOnce()
    expect(onDecline).toHaveBeenCalledOnce()
  })

  it('offers a question’s choices and a proposal’s two answers', () => {
    const onChoose = vi.fn()
    const onAccept = vi.fn()
    render(
      <>
        <QuestionCard question="Which Maya?" choices={[{ label: 'Maya Lin', detail: 'sent 2 albums' }, { label: 'Maya Ortiz' }]} onChoose={onChoose} />
        <ProposalCard text="Add the club night to your calendar." onAccept={onAccept} onDismiss={vi.fn()} />
      </>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Maya Lin · sent 2 albums' }))
    expect(onChoose).toHaveBeenCalledWith('Maya Lin')
    fireEvent.click(screen.getByRole('button', { name: 'Do it' }))
    expect(onAccept).toHaveBeenCalledOnce()
  })

  it('writes job progress as numbers as well as a bar', () => {
    render(<JobCard app="music" title="Sort the library" done={164} total={1480} />)
    expect(screen.getByText('running · 164 of 1,480')).toBeInTheDocument()
    expect(screen.getByRole('progressbar', { name: 'Sort the library progress' })).toHaveAttribute('value', '164')
  })

  it('renders the finished note, utterance, narration, ledger, side thread and memory fact', () => {
    render(
      <>
        <FinishedNote app="music" at="6:12 pm" text="Honey: 9 tracks." onOpen={vi.fn()} />
        <Utterance text="download the new Caribou album" followUp="Yes." />
        <Narration>Queued Honey.</Narration>
        <Ledger lines={[{ time: '18:04:06.3', event: 'plan.compiled' }, { time: '18:04:07.2', event: 'step 1 running' }]} />
        <SideThread app="chess" summary="1 call · timed out" calls={[{ name: 'next_club_night', outcome: 'no answer' }]} failed />
        <MemoryFact source="recall · your words" date="Tue Oct 27" fact="skip chess club until December" onForget={vi.fn()} />
      </>,
    )
    expect(screen.getByText('Finished · music · 6:12 pm')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Open in music' })).toBeInTheDocument()
    expect(screen.getByText('Yes.')).toBeInTheDocument()
    expect(screen.getByText('Queued Honey.')).toBeInTheDocument()
    expect(screen.getByText('step 1 running').closest('li')).toHaveAttribute('aria-current', 'true')
    expect(screen.getByText('1 call · timed out').closest('details')).toHaveAttribute('data-state', 'failed')
    expect(screen.getByText('skip chess club until December')).toBeInTheDocument()
  })

  describe('side threads (#139)', () => {
    it('heads an open thread with what it carries, outside titles quoted, and closes on Done', () => {
      const onDone = vi.fn()
      render(<ThreadHead title="Garage" carries={[{ label: 'Order brackets' }, { label: 'A web title', outside: true }]} onDone={onDone} onBack={vi.fn()} />)
      const head = screen.getByRole('region', { name: 'Side thread: Garage' })
      expect(head).toHaveTextContent('Carries')
      expect(screen.getByText('Order brackets')).toBeInTheDocument()
      expect(screen.getByText('“A web title”')).toHaveAttribute('data-outside', 'true')
      fireEvent.click(screen.getByRole('button', { name: 'Done · back to the timeline' }))
      expect(onDone).toHaveBeenCalledOnce()
    })

    it('says closing while the result is written, and closed only with the result', () => {
      const { rerender } = render(<ThreadHead title="Garage" carries={[]} closing onDone={vi.fn()} onBack={vi.fn()} />)
      expect(screen.getByRole('button', { name: 'Closing…' })).toBeDisabled()
      expect(screen.getByText('nothing: it started empty')).toBeInTheDocument()
      expect(screen.queryByText(/What flowed back/)).not.toBeInTheDocument()
      const onBack = vi.fn()
      rerender(<ThreadHead title="Garage" carries={[]} result="Added “Order brackets”." onBack={onBack} />)
      expect(screen.getByText('Closed. What flowed back: Added “Order brackets”.')).toBeInTheDocument()
      fireEvent.click(screen.getByRole('button', { name: 'Back to the timeline' }))
      expect(onBack).toHaveBeenCalledOnce()
    })

    it('shows a result note in the timeline that opens its thread', () => {
      const onOpen = vi.fn()
      render(<ThreadNote title="Garage" text="Added “Order brackets”." onOpen={onOpen} />)
      const note = screen.getByRole('group', { name: 'Side thread result: Garage' })
      expect(note).toHaveTextContent('side thread')
      expect(note).toHaveTextContent('Garage · Added “Order brackets”.')
      fireEvent.click(screen.getByRole('button', { name: 'Open' }))
      expect(onOpen).toHaveBeenCalledOnce()
    })

    it('tabs between the timeline and each open thread', () => {
      const onSelect = vi.fn()
      render(<ThreadTabs threads={[{ id: 7, title: 'Garage' }, { id: 8, title: 'Party' }]} active={8} onSelect={onSelect} />)
      expect(screen.getByRole('button', { name: 'Party' })).toHaveAttribute('aria-current', 'page')
      expect(screen.getByRole('button', { name: 'Timeline' })).not.toHaveAttribute('aria-current')
      fireEvent.click(screen.getByRole('button', { name: 'Timeline' }))
      expect(onSelect).toHaveBeenCalledWith(null)
    })

    it('asks for no name while incognito', () => {
      const onStart = vi.fn()
      render(<ThreadStart name="" incognito busy={false} onName={vi.fn()} onStart={onStart} onCancel={vi.fn()} />)
      expect(screen.queryByLabelText('Name (optional)')).not.toBeInTheDocument()
      expect(screen.getByText('Incognito is on, so its name isn’t kept.')).toBeInTheDocument()
      fireEvent.click(screen.getByRole('button', { name: 'Start' }))
      expect(onStart).toHaveBeenCalledOnce()
    })
  })
})
