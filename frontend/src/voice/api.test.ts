import { afterEach, describe, expect, it, vi } from 'vitest'
import { transcribe } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('transcribe', () => {
  it('returns the recognized text on success', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ text: 'hello' }), { status: 200 })),
    )
    await expect(transcribe(new Blob(['x']))).resolves.toBe('hello')
  })

  it('returns null on a non-2xx response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 503 })))
    await expect(transcribe(new Blob(['x']))).resolves.toBeNull()
  })

  it('returns null when fetch rejects (offline backend, DNS/proxy loss, abort)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))),
    )
    await expect(transcribe(new Blob(['x']))).resolves.toBeNull()
  })

  it('returns null when the body is not the expected JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('<html>proxy error', { status: 200 })))
    await expect(transcribe(new Blob(['x']))).resolves.toBeNull()
  })

  it('returns null when the JSON lacks a string text field', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ detail: 'nope' }), { status: 200 })),
    )
    await expect(transcribe(new Blob(['x']))).resolves.toBeNull()
  })

  it('gives up on a request that never answers, reporting unavailable', async () => {
    // #24: a backend or proxy that accepts the request and then never settles
    // it rejects nothing, so the catch never runs. Without a client deadline
    // the promise hangs, and MicButton stays in 'transcribing' — the state
    // that disables the button the user would tap to retry or exit.
    vi.useFakeTimers()
    const signals: (AbortSignal | undefined)[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init?: RequestInit) => {
        signals.push(init?.signal ?? undefined)
        return new Promise<Response>(() => {})
      }),
    )
    let settled = false
    const done = transcribe(new Blob(['x'])).then((text) => {
      settled = true
      return text
    })
    // Still waiting well past the backend's own 60s speech timeout: a slow but
    // living upstream must win the race and report its own failure.
    await vi.advanceTimersByTimeAsync(60_000)
    expect(settled).toBe(false)
    await vi.advanceTimersByTimeAsync(30_000)
    await expect(done).resolves.toBeNull()
    expect(settled).toBe(true)
    // The request itself is cancelled, not just abandoned.
    expect(signals[0]?.aborted).toBe(true)
  })

  it('sends the interaction id as X-Interaction-Id only when one is given', async () => {
    // The vendored MicButton passes whatever its optional beginUtterance hook
    // returned (chess's latency tracing); conductor passes no hook, so no header.
    const fetchMock = vi.fn<typeof fetch>(
      async () => new Response(JSON.stringify({ text: 'hello' }), { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)
    await transcribe(new Blob(['x']), 'clip.wav', 'a1b2c3d4e5f6')
    await transcribe(new Blob(['x']))
    expect(fetchMock.mock.calls[0][1]?.headers).toEqual({ 'X-Interaction-Id': 'a1b2c3d4e5f6' })
    expect(fetchMock.mock.calls[1][1]?.headers).toBeUndefined()
  })

  it('clears the deadline once the transcript has been read', async () => {
    // A completed request must not leave a timer armed to abort nothing.
    vi.useFakeTimers()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ text: 'hello' }), { status: 200 })),
    )
    await expect(transcribe(new Blob(['x']))).resolves.toBe('hello')
    expect(vi.getTimerCount()).toBe(0)
  })
})
