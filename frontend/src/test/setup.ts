import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// globals: true isn't set, so Testing Library's auto-cleanup never registers.
// Unmount between tests so rendered DOM doesn't leak across cases.
afterEach(cleanup)
