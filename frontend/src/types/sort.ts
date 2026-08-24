// Mirrors music's app/schemas/sorting.py — the worklist as the sort panel reads
// it. Fetched from music through conductor's fleet action proxy, never parsed
// out of a reply: what the panel shows is the state of the library, not a 12B's
// account of it.

/** One artist's group as the worklist lists it: a question, not an answer. */
export interface UnsortedGroup {
  artist: string
  tracks: number
  /** A few titles, enough to recognize the artist's output. */
  titles: string[]
  /** Leftover genre tags a ripper wrote. Shown to a person, never believed. */
  tags_say: string[]
}

/** A group opened up: every song waiting, because an artist is not one genre. */
export interface OpenedGroup {
  artist: string
  tracks: string[]
  tags_say: string[]
}

export interface SortStatus {
  filed_tracks: number
  filed_artist: string | null
  filed_into: string | null
  created_folder: boolean
  /** Titles that were named but are not waiting — reported, never dropped. */
  not_found: string[]
  opened: OpenedGroup | null

  unsorted_tracks: number
  unsorted_artists: number
  /** Every genre folder that exists. The whole vocabulary; there is no list. */
  folders: string[]
  next_up: UnsortedGroup[]
}

/** One answer, as the panel sends it. Empty `tracks` means the whole group. */
export interface Filing {
  artist: string
  genre: string
  tracks?: string[]
  correcting?: boolean
}
