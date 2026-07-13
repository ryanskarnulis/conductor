import { afterEach, describe, expect, it } from 'vitest'
import { handoffFrom } from './handoff'
import type { ToolCallRecord } from '../../types/agent'

const PAYLOAD = {
  handoff: 'chess',
  title: 'Chess',
  path: '/',
  upstream: '127.0.0.1:8000',
  intent_param: 'intent',
  intent: "let's play chess as black",
}

function call(overrides: Partial<ToolCallRecord> = {}): ToolCallRecord {
  return {
    tool: 'open_chess',
    arguments: { intent: "let's play chess as black" },
    result: JSON.stringify(PAYLOAD),
    error: null,
    ...overrides,
  }
}

/** jsdom's location is read-only; swap the whole object to serve the page from
 * a given host (LAN gateway vs. the dev server). */
function servedFrom(href: string) {
  Object.defineProperty(window, 'location', {
    value: new URL(href),
    writable: true,
    configurable: true,
  })
}

afterEach(() => servedFrom('http://localhost:3000/'))

describe('handoffFrom', () => {
  it('sends the user to the app on the host that served this page', () => {
    servedFrom('https://conductor.home.lan/c/7')

    const handoff = handoffFrom([call()])

    // Same rule as the gateway hub page: every app is <name>.<this host>.
    expect(handoff).toEqual({
      app: 'chess',
      title: 'Chess',
      url: "https://chess.home.lan/?intent=let%27s+play+chess+as+black",
    })
  })

  it('falls back to the upstream on a bare host (the dev server)', () => {
    servedFrom('http://localhost:5174/c/7')

    // localhost has no domain to graft `chess.` onto, so dial the app direct.
    expect(handoffFrom([call()])?.url).toBe(
      "http://127.0.0.1:8000/?intent=let%27s+play+chess+as+black",
    )
  })

  it('falls back to the upstream on a raw LAN IP', () => {
    // Dots, but no domain to swap a subdomain into — this host isn't behind
    // the gateway, so `chess.192.168` would be nonsense.
    servedFrom('http://192.168.1.20:8300/c/7')

    expect(handoffFrom([call()])?.url).toContain('http://127.0.0.1:8000/')
  })

  it('honors the app path and an absent intent', () => {
    servedFrom('https://conductor.home.lan/')
    const payload = { handoff: 'arcade', title: 'Arcade', path: '/play', upstream: '127.0.0.1:8500' }

    const handoff = handoffFrom([call({ tool: 'open_arcade', result: JSON.stringify(payload) })])

    // Nothing to carry → a bare URL, no dangling query string.
    expect(handoff?.url).toBe('https://arcade.home.lan/play')
  })

  it('is null when the turn asked for no handoff', () => {
    expect(handoffFrom(null)).toBeNull()
    expect(handoffFrom([])).toBeNull()
    expect(handoffFrom([call({ tool: 'ask_tasks', result: 'two tasks due' })])).toBeNull()
  })

  it('is null when the open call failed — a failed handoff navigates nowhere', () => {
    expect(handoffFrom([call({ result: null, error: 'chess is unreachable' })])).toBeNull()
  })

  it('is null when the payload is unreadable', () => {
    expect(handoffFrom([call({ result: 'not json' })])).toBeNull()
    expect(handoffFrom([call({ result: JSON.stringify({ nope: true }) })])).toBeNull()
  })
})
