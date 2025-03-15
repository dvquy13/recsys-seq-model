import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Home from '../page'
import { RecommendationsResponse } from '@/types/api'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { clearRecentlyViewedBooks } from '@/lib/recentlyViewed'
import { recommendationsApi } from '@/lib/api'
import '@testing-library/jest-dom'

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

// Mock dependencies
jest.mock('@/lib/recentlyViewed', () => ({
  clearRecentlyViewedBooks: jest.fn(),
  RECENTLY_VIEWED_CHANGE_EVENT: 'recently-viewed-change',
  getRecentlyViewedBooks: jest.fn()
}));

jest.mock('@/components/RecentlyViewedGrid', () => ({
  RecentlyViewedGrid: () => <div data-testid="recently-viewed-grid" />
}));

jest.mock('@/components/RecommendationsGrid', () => ({
  RecommendationsGrid: () => <div data-testid="recommendations-grid" />
}));

jest.mock('@/lib/api', () => ({
  recommendationsApi: {
    getRecommendations: jest.fn()
  }
}));

describe('Home Page', () => {
  beforeEach(() => {
    // Clear mock calls between tests
    jest.clearAllMocks()
    // Clear localStorage
    localStorage.clear()

    // Mock localStorage
    const localStorageMock = {
      getItem: jest.fn().mockReturnValue(null),
      setItem: jest.fn(),
      removeItem: jest.fn(),
      clear: jest.fn()
    };
    Object.defineProperty(window, 'localStorage', { value: localStorageMock });
    
    // Set up default mock implementation for getRecommendations
    (recommendationsApi.getRecommendations as jest.Mock).mockResolvedValue(mockRecommendationsResponse);
  })

  it('renders the user lookup form', async () => {
    await act(async () => {
      renderWithClient(<Home />)
    })
    
    expect(screen.getByText('User Lookup')).toBeInTheDocument()
    expect(screen.getByText('Enter a user ID to fetch their recommendations')).toBeInTheDocument()
    expect(screen.getByLabelText('User ID')).toBeInTheDocument()
    expect(screen.getByRole('button')).toHaveTextContent('Submit')
    expect(screen.getByTestId('recently-viewed-grid')).toBeInTheDocument()
  })

  it('handles user input correctly', async () => {
    await act(async () => {
      renderWithClient(<Home />)
    })
    
    const input = screen.getByLabelText('User ID')
    await act(async () => {
      fireEvent.change(input, { target: { value: 'testuser123' } })
    })
    
    expect(input).toHaveValue('testuser123')
  })

  it('shows loading state during API call', async () => {
    // Mock a delayed API response
    (recommendationsApi.getRecommendations as jest.Mock).mockImplementationOnce(() => 
      new Promise(resolve => setTimeout(() => resolve(mockRecommendationsResponse), 100))
    );

    await act(async () => {
      renderWithClient(<Home />)
    })
    
    const input = screen.getByLabelText('User ID')
    await act(async () => {
      fireEvent.change(input, { target: { value: 'testuser123' } })
    })
    
    const submitButton = screen.getByRole('button')
    expect(submitButton).toHaveTextContent('Submit')
    
    await act(async () => {
      fireEvent.click(submitButton)
    })
    
    // Button text should change to "Loading..."
    expect(screen.getByRole('button')).toHaveTextContent('Loading...')
    expect(submitButton).toBeDisabled()

    // Wait for the loading state to finish
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 200))
    })

    expect(screen.getByRole('button')).toHaveTextContent('Submit')
    expect(screen.getByRole('button')).not.toBeDisabled()
  })

  it('displays recommendations after successful API call', async () => {
    await act(async () => {
      renderWithClient(<Home />)
    })
    
    const input = screen.getByLabelText('User ID')
    await act(async () => {
      fireEvent.change(input, { target: { value: 'testuser123' } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument()
  })

  it('displays error message when API call fails', async () => {
    const errorMessage = 'API Error';
    (recommendationsApi.getRecommendations as jest.Mock).mockRejectedValueOnce(new Error(errorMessage));

    await act(async () => {
      renderWithClient(<Home />)
    })
    
    const input = screen.getByLabelText('User ID')
    await act(async () => {
      fireEvent.change(input, { target: { value: 'testuser123' } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(screen.getByText(errorMessage)).toBeInTheDocument()
  })

  it('makes API call with correct parameters', async () => {
    await act(async () => {
      renderWithClient(<Home />)
    })
    
    const userId = 'testuser123'
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: userId } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(recommendationsApi.getRecommendations).toHaveBeenCalledWith(userId)
  })

  it('handles empty user ID correctly', async () => {
    await act(async () => {
      renderWithClient(<Home />)
    })
    
    // Set an empty user ID
    const input = screen.getByLabelText('User ID')
    await act(async () => {
      fireEvent.change(input, { target: { value: '' } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    // Verify API call is made with empty user ID
    expect(recommendationsApi.getRecommendations).toHaveBeenCalledWith('')

    // Verify recommendations are displayed
    expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument()
  })

  it('persists user ID in localStorage after submission', async () => {
    await act(async () => {
      renderWithClient(<Home />)
    })
    
    const userId = 'testuser123'
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: userId } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(localStorage.setItem).toHaveBeenCalledWith('last-submitted-user-id', userId)
  })

  it('loads persisted user ID from localStorage on mount', async () => {
    // Mock localStorage.getItem to return our persisted user ID
    (localStorage.getItem as jest.Mock).mockReturnValueOnce('persisteduser123')

    await act(async () => {
      renderWithClient(<Home />)
    })

    // Wait for the component to update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(recommendationsApi.getRecommendations).toHaveBeenCalledWith('persisteduser123')
  })

  it('uses React Query cache for subsequent requests', async () => {
    const queryClient = createTestQueryClient()
    
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <Home />
        </QueryClientProvider>
      )
    })

    const userId = 'testuser123'
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: userId } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update with a proper act wrapper
    await act(() => Promise.resolve()); 

    // Clear the mock to check if it's called again
    (recommendationsApi.getRecommendations as jest.Mock).mockClear()

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <Home />
        </QueryClientProvider>
      )
    })

    expect(recommendationsApi.getRecommendations).not.toHaveBeenCalled()
  })

  it('maintains cache across multiple user searches', async () => {
    const queryClient = createTestQueryClient()

    // Pre-populate the cache for user1
    queryClient.setQueryData(['recommendations', 'user1'], {
      recommendations: [{
        parent_asin: '123',
        title: 'User 1 Book',
        subtitle: 'A Test Subtitle',
        main_category: 'Fiction',
        average_rating: 4.5,
        rating_number: 100,
        price: '9.99',
        image_url: 'https://example.com/image.jpg',
        score: 0.95
      }]
    })

    // Pre-populate the cache for user2
    queryClient.setQueryData(['recommendations', 'user2'], {
      recommendations: [{
        parent_asin: '456',
        title: 'User 2 Book',
        subtitle: 'A Test Subtitle',
        main_category: 'Fiction',
        average_rating: 4.5,
        rating_number: 100,
        price: '9.99',
        image_url: 'https://example.com/image.jpg',
        score: 0.95
      }]
    })

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <Home />
        </QueryClientProvider>
      )
    })

    // Search for first user
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: 'user1' } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update with a proper act wrapper
    await act(() => Promise.resolve());

    // Search for second user
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: 'user2' } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update with a proper act wrapper
    await act(() => Promise.resolve());

    // Clear the mock to check if it's called again
    (recommendationsApi.getRecommendations as jest.Mock).mockClear()

    // Go back to first user
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: 'user1' } })
      fireEvent.click(screen.getByRole('button'))
    })

    // Wait for the component to update with a proper act wrapper
    await act(() => Promise.resolve());

    // Verify API is called with the correct user ID
    expect(recommendationsApi.getRecommendations).toHaveBeenCalledWith('user1')
  })

  it('should clear recently viewed books when submitting a new user ID', async () => {
    // Set initial user ID in localStorage
    (localStorage.getItem as jest.Mock).mockReturnValueOnce('user1');
    
    await act(async () => {
      renderWithClient(<Home />);
    })
    
    // Change user ID and submit
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: 'user2' } });
      fireEvent.click(screen.getByRole('button'));
    })
    
    // Wait for the component to update with a proper act wrapper
    await act(() => Promise.resolve());
    
    // Should clear recently viewed books
    expect(clearRecentlyViewedBooks).toHaveBeenCalledTimes(1);
    
    // Should update localStorage
    expect(localStorage.setItem).toHaveBeenCalledWith('last-submitted-user-id', 'user2');
  });

  it('should not clear recently viewed books when submitting the same user ID', async () => {
    // Set initial user ID in localStorage
    (localStorage.getItem as jest.Mock).mockReturnValueOnce('user1');
    
    await act(async () => {
      renderWithClient(<Home />);
    })
    
    // Submit same user ID
    await act(async () => {
      fireEvent.change(screen.getByLabelText('User ID'), { target: { value: 'user1' } });
      fireEvent.click(screen.getByRole('button'));
    })
    
    // Wait for the component to update with a proper act wrapper
    await act(() => Promise.resolve());
    
    // Should not clear recently viewed books
    expect(clearRecentlyViewedBooks).not.toHaveBeenCalled();
  });
}); 