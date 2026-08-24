import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fileSongs, getWorklist, openGroup } from '../../api/sort'
import type { SortStatus } from '../../types/sort'
import { SortPanel } from './SortPanel'

vi.mock('../../api/sort', () => ({
  getWorklist: vi.fn(),
  openGroup: vi.fn(),
  fileSongs: vi.fn(),
}))

const mockWorklist = vi.mocked(getWorklist)
const mockOpen = vi.mocked(openGroup)
const mockFile = vi.mocked(fileSongs)

function status(overrides: Partial<SortStatus> = {}): SortStatus {
  return {
    filed_tracks: 0,
    filed_artist: null,
    filed_into: null,
    created_folder: false,
    not_found: [],
    opened: null,
    unsorted_tracks: 7,
    unsorted_artists: 2,
    folders: ['Dubstep', 'House', 'Other'],
    next_up: [
      { artist: 'Zeds Dead', tracks: 5, titles: ['Collapse', 'Rumble'], tags_say: ['Bass'] },
      { artist: 'Matroda', tracks: 2, titles: ['Get Down'], tags_say: [] },
    ],
    ...overrides,
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  mockWorklist.mockResolvedValue(status())
})

function panel(onFiled = vi.fn()) {
  render(<SortPanel onFiled={onFiled} onDismiss={vi.fn()} />)
  return onFiled
}

describe('SortPanel', () => {
  it('asks about the biggest group first, with the folders that exist', async () => {
    panel()

    expect(await screen.findByText('Zeds Dead')).toBeInTheDocument()
    expect(screen.getByText('5 songs')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Dubstep' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'House' })).toBeInTheDocument()
  })

  it('shows the leftover tag as a hint, never as an answer', async () => {
    panel()
    expect(await screen.findByText(/tags say: Bass/)).toBeInTheDocument()
  })

  it('files the whole group on a click, and records it in the thread', async () => {
    mockFile.mockResolvedValue(
      status({ filed_tracks: 5, filed_artist: 'Zeds Dead', filed_into: 'Dubstep' }),
    )
    const onFiled = panel()

    fireEvent.click(await screen.findByRole('button', { name: 'Dubstep' }))

    await waitFor(() =>
      expect(mockFile).toHaveBeenCalledWith({ artist: 'Zeds Dead', genre: 'Dubstep', tracks: [] }),
    )
    expect(onFiled).toHaveBeenCalledWith('Zeds Dead (5 songs) → Dubstep')
  })

  it('opens a group up so one artist can go to more than one folder', async () => {
    mockOpen.mockResolvedValue(
      status({
        opened: { artist: 'Zeds Dead', tracks: ['Collapse', 'Rumble', 'Blackout'], tags_say: [] },
      }),
    )
    mockFile.mockResolvedValue(status({ filed_tracks: 1, filed_into: 'House' }))
    const onFiled = panel()

    fireEvent.click(await screen.findByRole('button', { name: /One at a time/ }))
    const collapse = await screen.findByLabelText('Collapse')
    fireEvent.click(collapse)
    fireEvent.click(screen.getByRole('button', { name: 'House' }))

    await waitFor(() =>
      expect(mockFile).toHaveBeenCalledWith({
        artist: 'Zeds Dead',
        genre: 'House',
        tracks: ['Collapse'],
      }),
    )
    expect(onFiled).toHaveBeenCalledWith('Zeds Dead: Collapse → House')
  })

  it('will not file an opened group until something is ticked', async () => {
    mockOpen.mockResolvedValue(
      status({ opened: { artist: 'Zeds Dead', tracks: ['Collapse'], tags_say: [] } }),
    )
    panel()

    fireEvent.click(await screen.findByRole('button', { name: /One at a time/ }))

    await waitFor(() => expect(screen.getByRole('button', { name: 'House' })).toBeDisabled())
    expect(mockFile).not.toHaveBeenCalled()
  })

  it('skips to the next artist without writing anything', async () => {
    panel()

    fireEvent.click(await screen.findByRole('button', { name: /Skip/ }))

    expect(await screen.findByText('Matroda')).toBeInTheDocument()
    expect(mockFile).not.toHaveBeenCalled()
  })

  it('files into a folder that does not exist yet', async () => {
    mockFile.mockResolvedValue(status({ filed_tracks: 5, created_folder: true }))
    panel()

    fireEvent.click(await screen.findByRole('button', { name: /New folder/ }))
    fireEvent.change(screen.getByLabelText('New folder name'), { target: { value: 'Drum & Bass' } })
    fireEvent.click(screen.getByRole('button', { name: /Create & file/ }))

    await waitFor(() =>
      expect(mockFile).toHaveBeenCalledWith({
        artist: 'Zeds Dead',
        genre: 'Drum & Bass',
        tracks: [],
      }),
    )
  })

  it("reports the app's own refusal instead of guessing at it", async () => {
    const { ApiError } = await import('../../api/client')
    mockFile.mockRejectedValue(new ApiError(422, { detail: 'ask again as a correction' }))
    const onFiled = panel()

    fireEvent.click(await screen.findByRole('button', { name: 'Dubstep' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('ask again as a correction')
    expect(onFiled).not.toHaveBeenCalled()
  })

  it('says so when nothing is left to sort', async () => {
    mockWorklist.mockResolvedValue(status({ unsorted_tracks: 0, unsorted_artists: 0, next_up: [] }))
    panel()

    expect(await screen.findByText(/Nothing left waiting/)).toBeInTheDocument()
  })
})
