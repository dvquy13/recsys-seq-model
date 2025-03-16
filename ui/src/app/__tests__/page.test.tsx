import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Home from '../page'
import { RecommendationsResponse } from '@/types/api'
import * as AppStateContext from '@/providers/app-state-provider'
import '@testing-library/jest-dom'

// Mock the AppState context
jest.mock('@/providers/app-state-provider', () => ({
  useAppState: jest.fn()
}))

// Mock components that render recommendations
jest.mock('@/components/RecentlyViewedGrid', () => ({
  RecentlyViewedGrid: () => <div data-testid="recently-viewed-grid" />
}))

jest.mock('@/components/RecommendationsGrid', () => ({
  RecommendationsGrid: ({ title, recommendations }: any) => (
    <div data-testid="recommendations-grid">
      <div data-testid="recommendations-title">{title}</div>
      <div data-testid="recommendations-count">{recommendations?.length || 0}</div>
    </div>
  )
}))

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
  // Mock functions to test state changes
  const mockSetUserId = jest.fn()
  const mockSubmitUserId = jest.fn()
  
  beforeEach(() => {
    // Clear mock calls between tests
    jest.clearAllMocks()
    
    // Setup the mock implementation for useAppState with default values
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      userId: '',
      setUserId: mockSetUserId,
      submittedUserId: null,
      submitUserId: mockSubmitUserId,
      recentlyViewedBooks: [],
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: false,
      recommendationsError: null
    })
  })

  it('renders the user lookup form', async () => {
    render(<Home />)
    
    expect(screen.getByText('User Lookup')).toBeInTheDocument()
    expect(screen.getByText('Enter a user ID to fetch their recommendations')).toBeInTheDocument()
    expect(screen.getByLabelText('User ID')).toBeInTheDocument()
    expect(screen.getByRole('button')).toHaveTextContent('Submit')
    expect(screen.getByTestId('recently-viewed-grid')).toBeInTheDocument()
  })

  it('handles user input correctly', async () => {
    render(<Home />)
    
    const input = screen.getByLabelText('User ID')
    await act(async () => {
      fireEvent.change(input, { target: { value: 'testuser123' } })
    })
    
    expect(mockSetUserId).toHaveBeenCalledWith('testuser123')
  })

  it('shows loading state during API call', async () => {
    // Mock loading state
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      userId: 'testuser123',
      setUserId: mockSetUserId,
      submittedUserId: null,
      submitUserId: mockSubmitUserId,
      recentlyViewedBooks: [],
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: true, // Set loading to true
      recommendationsError: null
    })

    render(<Home />)
    
    // Button should show loading state
    expect(screen.getByRole('button')).toHaveTextContent('Loading...')
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('displays recommendations after successful API call', async () => {
    // Mock successful recommendations
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      userId: 'testuser123',
      setUserId: mockSetUserId,
      submittedUserId: 'testuser123',
      submitUserId: mockSubmitUserId,
      recentlyViewedBooks: [],
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: mockRecommendationsResponse, // Set recommendations
      isLoadingRecommendations: false,
      recommendationsError: null
    })

    render(<Home />)

    // Should display recommendations
    expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument()
    expect(screen.getByTestId('recommendations-title')).toHaveTextContent('Recommendations')
    expect(screen.getByTestId('recommendations-count')).toHaveTextContent('1')
  })

  it('displays error message when API call fails', async () => {
    const errorMessage = 'API Error'
    
    // Mock error state
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      userId: 'testuser123',
      setUserId: mockSetUserId,
      submittedUserId: 'testuser123',
      submitUserId: mockSubmitUserId,
      recentlyViewedBooks: [],
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: false,
      recommendationsError: new Error(errorMessage) // Set error
    })

    render(<Home />)

    // Should display the error message
    expect(screen.getByText(errorMessage)).toBeInTheDocument()
  })

  it('submits user ID when form is submitted', async () => {
    // Set up userId
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      userId: 'testuser123',
      setUserId: mockSetUserId,
      submittedUserId: null,
      submitUserId: mockSubmitUserId,
      recentlyViewedBooks: [],
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: false,
      recommendationsError: null
    })

    render(<Home />)
    
    // Submit using the button instead of the form
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }))
    })

    // Should call submitUserId with the user ID
    expect(mockSubmitUserId).toHaveBeenCalledWith('testuser123')
  })

  it('displays personalized recommendations when available', async () => {
    // Mock personalized recommendations
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      userId: 'testuser123',
      setUserId: mockSetUserId,
      submittedUserId: 'testuser123',
      submitUserId: mockSubmitUserId,
      recentlyViewedBooks: [],
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: mockRecommendationsResponse, // Set personalized recommendations
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: false,
      recommendationsError: null
    })

    render(<Home />)

    // Should display personalized recommendations
    expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument()
    expect(screen.getByTestId('recommendations-title')).toHaveTextContent('Recommendations Based on Your Browsing')
  })
}) 