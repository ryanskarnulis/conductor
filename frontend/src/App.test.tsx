import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('App', () => {
  it('renders the Conductor title', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Conductor' })).toBeInTheDocument()
  })

  it('renders the placeholder tagline', () => {
    render(<App />)
    expect(screen.getByText(/master agent/i)).toBeInTheDocument()
  })
})
