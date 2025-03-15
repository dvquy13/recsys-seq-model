import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Home from '../page'
import { RecommendationsResponse } from '@/types/api'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Mock fetch globally
global.fetch = jest.fn()

// Helper to create mock response
const createMockResponse = (data: any) => {
  return { ok: true, json: () => Promise.resolve(data) }
}

// Create a new QueryClient for each test
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
})

// Custom render function that includes QueryClientProvider
const renderWithClient = (ui: React.ReactElement) => {
  const testQueryClient = createTestQueryClient()
  return render(
    <QueryClientProvider client={testQueryClient}>
      {ui}
    </QueryClientProvider>
  )
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
    // Clear localStorage
    localStorage.clear()
  })

  it('renders the user lookup form', () => {
    renderWithClient(<Home />)
    
    expect(screen.getByText('User Lookup')).toBeInTheDocument()
    expect(screen.getByText('Enter a user ID to fetch their recommendations')).toBeInTheDocument()
    expect(screen.getByLabelText('User ID')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument()
  })

  it('handles user input correctly', () => {
    renderWithClient(<Home />)
    
    const input = screen.getByLabelText('User ID')
    fireEvent.change(input, { target: { value: 'testuser123' } })
    
    expect(input).toHaveValue('testuser123')
  })

  it('shows loading state during API call', async () => {
    // Mock fetch to return a delayed response
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      new Promise(resolve => setTimeout(() => resolve(createMockResponse(mockRecommendationsResponse)), 100))
    )

    renderWithClient(<Home />)
    
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

    renderWithClient(<Home />)
    
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

    renderWithClient(<Home />)
    
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

    renderWithClient(<Home />)
    
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

  it('handles empty user ID correctly', async () => {
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      Promise.resolve(createMockResponse(mockRecommendationsResponse))
    )

    renderWithClient(<Home />)
    
    // Set an empty user ID
    const input = screen.getByLabelText('User ID')
    fireEvent.change(input, { target: { value: '' } })
    
    // Submit the form
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => {
      // Verify API call is made with empty user ID
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/recs/retrieve?count=10&debug=false',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'accept': 'application/json',
          },
          body: JSON.stringify({
            user_ids_raw: [''],
            item_seq_raw: [[]],
            candidate_items_raw: []
          })
        })
      )

      // Verify recommendations are displayed
      expect(screen.getByText('Recommendations')).toBeInTheDocument()
      expect(screen.getByText('Test Book')).toBeInTheDocument()
    })
  })
}) 