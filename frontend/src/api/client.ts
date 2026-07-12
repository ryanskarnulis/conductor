export class ApiError extends Error {
  readonly status: number
  readonly body: unknown

  constructor(status: number, body: unknown) {
    super(`API error ${status}`)
    this.status = status
    this.body = body
  }
}

export class ApiTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`The request timed out after ${Math.round(timeoutMs / 1000)}s — is the backend running?`)
  }
}

const BASE_URL =
  import.meta.env.VITE_API_URL ?? `http://${window.location.hostname}:8301`

// Every request aborts eventually so a hung backend can't leave a permanent
// spinner.
const DEFAULT_TIMEOUT_MS = 30_000

export interface ApiOptions extends RequestInit {
  timeoutMs?: number
}

export async function apiClient(
  path: string,
  options?: ApiOptions,
): Promise<Response> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...init } = options ?? {}
  // A caller-provided signal wins (it already owns cancellation).
  const signal = init.signal ?? AbortSignal.timeout(timeoutMs)
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, { ...init, signal })
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === 'TimeoutError') {
      throw new ApiTimeoutError(timeoutMs)
    }
    throw e
  }
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    throw new ApiError(response.status, body)
  }
  return response
}
