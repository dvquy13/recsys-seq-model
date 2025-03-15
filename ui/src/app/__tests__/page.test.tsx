import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Home from '../page'

describe('Home Component', () => {
  it('renders the form with all elements', () => {
    render(<Home />)
    
    // Check if title and description are present
    expect(screen.getByText('User Lookup')).toBeInTheDocument()
    expect(screen.getByText('Enter a user ID to fetch their information')).toBeInTheDocument()
    
    // Check if input and button are present
    expect(screen.getByLabelText('User ID')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument()
  })

  it('allows entering a user ID', async () => {
    render(<Home />)
    const input = screen.getByLabelText('User ID')
    
    await userEvent.type(input, '12345')
    expect(input).toHaveValue('12345')
  })

  it('handles form submission', async () => {
    const consoleSpy = jest.spyOn(console, 'log')
    render(<Home />)
    
    const input = screen.getByLabelText('User ID')
    const submitButton = screen.getByRole('button', { name: 'Submit' })
    
    await userEvent.type(input, '12345')
    await userEvent.click(submitButton)
    
    expect(consoleSpy).toHaveBeenCalledWith('Submitted user ID:', '12345')
    consoleSpy.mockRestore()
  })

  it('requires user ID input', () => {
    render(<Home />)
    const input = screen.getByLabelText('User ID')
    expect(input).toBeRequired()
  })
}) 