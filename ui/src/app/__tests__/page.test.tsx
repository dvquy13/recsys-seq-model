import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Home from '../page'
import { RecommendationsResponse } from '@/types/api'

// Mock DiceBear modules
jest.mock('@dicebear/core', () => ({
  createAvatar: jest.fn().mockImplementation(() => ({
    toDataUri: () => 'data:image/svg+xml;charset=UTF-8,<svg><rect fill="#ff0000"/><ellipse/><polygon/></svg>'
  }))
}));

jest.mock('@dicebear/collection', () => ({
  shapes: {}
}));

// Mock fetch globally
global.fetch = jest.fn()

// Helper to create mock response
const createMockResponse = (data: any) => {
  return { ok: true, json: () => Promise.resolve(data) }
}

const mockRecommendationsResponse: RecommendationsResponse = {
  recommendations: [
    {
      score: 0.6588092,
      main_category: "Books",
      title: "Test Book",
      average_rating: 4.5,
      rating_number: 100,
      price: "9.99",
      subtitle: "Kindle Edition",
      image_url: "https://example.com/image.jpg",
      parent_asin: "B000Q9INGI"
    }
  ],
  ctx: {
    user_ids_raw: ["testuser123"],
    item_seq_raw: [[]],
    candidate_items_raw: []
  },
  metadata: {
    rec_id: "test-123"
  }
}

describe('Home Page', () => {
  beforeEach(() => {
    // Clear mock calls between tests
    jest.clearAllMocks()
  })

  it('renders the user lookup form', () => {
    render(<Home />)
    
    expect(screen.getByText('User Lookup')).toBeInTheDocument()
    expect(screen.getByText('Enter a user ID to fetch their recommendations')).toBeInTheDocument()
    expect(screen.getByLabelText('User ID')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument()
  })

  it('handles user input correctly', () => {
    render(<Home />)
    
    const input = screen.getByLabelText('User ID')
    fireEvent.change(input, { target: { value: 'testuser123' } })
    
    expect(input).toHaveValue('testuser123')
  })

  it('shows loading state during API call', async () => {
    // Mock fetch to return a delayed response
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      new Promise(resolve => setTimeout(() => resolve(createMockResponse(mockRecommendationsResponse)), 100))
    )

    render(<Home />)
    
    const input = screen.getByLabelText('User ID')
    fireEvent.change(input, { target: { value: 'testuser123' } })
    
    const submitButton = screen.getByRole('button')
    fireEvent.click(submitButton)
    
    expect(screen.getByText('Loading...')).toBeInTheDocument()
    expect(submitButton).toBeDisabled()

    await waitFor(() => {
      expect(screen.queryByText('Loading...')).not.toBeInTheDocument()
      expect(submitButton).not.toBeDisabled()
    })
  })

  it('displays recommendations after successful API call', async () => {
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      Promise.resolve(createMockResponse(mockRecommendationsResponse))
    )

    render(<Home />)
    
    const input = screen.getByLabelText('User ID')
    fireEvent.change(input, { target: { value: 'testuser123' } })
    
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => {
      expect(screen.getByText('Recommendations')).toBeInTheDocument()
      expect(screen.getByText('Test Book')).toBeInTheDocument()
      expect(screen.getByText('Rating: 4.5 (100 reviews)')).toBeInTheDocument()
      expect(screen.getByText('Price: $9.99')).toBeInTheDocument()
    })
  })

  it('displays error message when API call fails', async () => {
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      Promise.reject(new Error('API Error'))
    )

    render(<Home />)
    
    const input = screen.getByLabelText('User ID')
    fireEvent.change(input, { target: { value: 'testuser123' } })
    
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => {
      expect(screen.getByText('API Error')).toBeInTheDocument()
    })
  })

  it('makes API call with correct parameters', async () => {
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      Promise.resolve(createMockResponse(mockRecommendationsResponse))
    )

    render(<Home />)
    
    const userId = 'testuser123'
    fireEvent.change(screen.getByLabelText('User ID'), { target: { value: userId } })
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/recs/retrieve?count=10&debug=false',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'accept': 'application/json',
          },
          body: JSON.stringify({
            user_ids_raw: [userId],
            item_seq_raw: [[]],
            candidate_items_raw: []
          })
        })
      )
    })
  })
}) 