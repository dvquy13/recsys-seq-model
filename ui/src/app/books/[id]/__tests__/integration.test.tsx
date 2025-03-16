import React from 'react';
import { render, screen } from '@testing-library/react';
import BookPage from '../page';
import { getBookDetails } from '@/lib/api';
import { recommendationsApi } from '@/lib/api';
import '@testing-library/jest-dom';

// Mock Next.js navigation
jest.mock('next/navigation', () => ({
  notFound: jest.fn(),
}));

// Mock the API calls - Make sure to properly mock recommendationsApi
jest.mock('@/lib/api', () => {
  return {
    getBookDetails: jest.fn(),
    recommendationsApi: {
      getRecommendations: jest.fn().mockResolvedValue({
        recommendations: [],
        ctx: { user_ids_raw: [], item_seq_raw: [[]], candidate_items_raw: [] },
        metadata: { rec_id: '123' }
      }),
    },
  };
});

// Mock the lucide-react icons
jest.mock('lucide-react', () => ({
  Star: () => <span data-testid="star-icon">★</span>,
  StarHalf: () => <span data-testid="star-half-icon">★</span>,
}));

// Mock components
jest.mock('@/components/book-covers', () => ({
  BookCover: () => <div data-testid="book-cover">Book Cover</div>,
}));

jest.mock('@/components/ui/book-rating', () => ({
  BookRating: ({ rating, ratingCount }: { rating: number, ratingCount: number }) => (
    <div data-testid="book-rating">
      <span>{rating.toFixed(1)}</span>
      <span>({ratingCount.toLocaleString()} ratings)</span>
    </div>
  ),
}));

jest.mock('@/components/ui/book-price', () => ({
  BookPrice: () => <div data-testid="book-price">Book Price</div>,
}));

// Mock the BookViewTracker to avoid API calls during testing
jest.mock('@/components/BookViewTracker', () => ({
  BookViewTracker: () => null,
}));

jest.mock('@/lib/config', () => ({
  getConfig: jest.fn().mockReturnValue('default'),
}));

// Properly mock the SimilarBooksSection
jest.mock('@/components/SimilarBooksSection', () => ({
  SimilarBooksSection: ({ bookId }: { bookId: string }) => (
    <div data-testid="similar-books-section">
      Similar Books Section for {bookId}
    </div>
  ),
}));

// Mock personalizedRecs module to prevent errors
jest.mock('@/lib/personalizedRecs', () => ({
  getRecommendations: jest.fn().mockResolvedValue({ recommendations: [] }),
  updatePersonalizedRecs: jest.fn().mockResolvedValue(true),
  getCachedPersonalizedRecs: jest.fn().mockReturnValue(null),
  hasPersonalizedRecs: jest.fn().mockReturnValue(false),
  clearPersonalizedRecs: jest.fn(),
  PERSONALIZED_RECS_UPDATED_EVENT: 'personalized-recs-updated',
}));

describe('BookPage Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const mockBook = {
    parent_asin: 'B001234567',
    asin: 'B001234567',
    title: 'Test Book',
    subtitle: 'A Test Book for Testing',
    average_rating: 4.5,
    rating_number: 1000,
    price: '9.99',
    main_category: 'Fiction',
    image_url: 'http://example.com/image.jpg',
  };

  it('should render the complete book page with similar books section', async () => {
    (getBookDetails as jest.Mock).mockResolvedValue(mockBook);
    
    // Using async render for Server Component
    const component = await BookPage({ params: { id: 'B001234567' } });
    render(component);
    
    // Check that all components are rendered
    expect(screen.getByText('Test Book')).toBeInTheDocument();
    expect(screen.getByTestId('book-cover')).toBeInTheDocument();
    expect(screen.getByTestId('book-rating')).toBeInTheDocument();
    expect(screen.getByTestId('book-price')).toBeInTheDocument();
    
    // Check that similar books section is rendered with correct bookId
    expect(screen.getByTestId('similar-books-section')).toBeInTheDocument();
    expect(screen.getByTestId('similar-books-section')).toHaveTextContent('Similar Books Section for B001234567');
  });

  it('should use parent_asin for similar books when available', async () => {
    const bookWithParentAsin = {
      ...mockBook,
      parent_asin: 'B00PARENT',
      asin: 'B00CHILD',
    };
    
    (getBookDetails as jest.Mock).mockResolvedValue(bookWithParentAsin);
    
    // Using async render for Server Component
    const component = await BookPage({ params: { id: 'B00CHILD' } });
    render(component);
    
    expect(screen.getByTestId('similar-books-section')).toHaveTextContent('Similar Books Section for B00PARENT');
  });

  it('should fall back to ID param if no parent_asin or asin is available', async () => {
    const bookWithoutAsins = {
      ...mockBook,
      parent_asin: undefined,
      asin: undefined,
    };
    
    (getBookDetails as jest.Mock).mockResolvedValue(bookWithoutAsins);
    
    // Using async render for Server Component
    const component = await BookPage({ params: { id: 'FALLBACK_ID' } });
    render(component);
    
    expect(screen.getByTestId('similar-books-section')).toHaveTextContent('Similar Books Section for FALLBACK_ID');
  });
}); 