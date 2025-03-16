import React from 'react';
import { render, screen } from '@testing-library/react';
import BookPage, { generateMetadata } from '../page';
import { getBookDetails } from '@/lib/api';
import { notFound } from 'next/navigation';

// Mock the next/navigation module
jest.mock('next/navigation', () => ({
  notFound: jest.fn(),
}));

// Mock the api module
jest.mock('@/lib/api', () => ({
  getBookDetails: jest.fn(),
  recommendationsApi: {
    getRecommendations: jest.fn().mockResolvedValue({
      recommendations: [],
      ctx: { user_ids_raw: [], item_seq_raw: [[]], candidate_items_raw: [] },
      metadata: { rec_id: 'test-123' }
    }),
  },
}));

// Mock the config module
jest.mock('@/lib/config', () => ({
  getConfig: jest.fn(() => 'textBased'),
}));

// Mock the lucide-react icons
jest.mock('lucide-react', () => ({
  Star: () => <span data-testid="star-icon">★</span>,
  StarHalf: () => <span data-testid="star-half-icon">★</span>,
}));

// Mock the BookViewTracker component
jest.mock('@/components/BookViewTracker', () => ({
  BookViewTracker: () => null,
}));

// Mock the SimilarBooksSection component
jest.mock('@/components/SimilarBooksSection', () => ({
  SimilarBooksSection: ({ bookId }: { bookId: string }) => (
    <div data-testid="similar-books-section">Similar Books for {bookId}</div>
  ),
}));

// Mock personalizedRecs to prevent errors
jest.mock('@/lib/personalizedRecs', () => ({
  getRecommendations: jest.fn().mockResolvedValue({ recommendations: [] }),
  updatePersonalizedRecs: jest.fn().mockResolvedValue(true),
  getCachedPersonalizedRecs: jest.fn().mockReturnValue(null),
  hasPersonalizedRecs: jest.fn().mockReturnValue(false),
  clearPersonalizedRecs: jest.fn(),
  PERSONALIZED_RECS_UPDATED_EVENT: 'personalized-recs-updated',
}));

// Mock BookRating component
jest.mock('@/components/ui/book-rating', () => ({
  BookRating: ({ rating, ratingCount }: { rating: number, ratingCount: number }) => (
    <div data-testid="book-rating">
      <span>{rating.toFixed(1)}</span>
      <span>({ratingCount.toLocaleString()} ratings)</span>
    </div>
  ),
}));

describe('BookPage', () => {
  const mockBook = {
    title: 'Test Book',
    subtitle: 'A Test Subtitle',
    average_rating: 4.5,
    rating_number: 1000,
    price: '9.99',
    main_category: 'Fiction',
    image_url: 'https://example.com/book.jpg',
    parent_asin: 'B000ASPUES',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (getBookDetails as jest.Mock).mockResolvedValue(mockBook);
  });

  describe('Page Component', () => {
    it('renders book details correctly', async () => {
      render(await BookPage({ params: { id: mockBook.parent_asin } }));

      // Check if main book information is rendered
      expect(screen.getByText(mockBook.title)).toBeInTheDocument();
      expect(screen.getByText(mockBook.subtitle)).toBeInTheDocument();
      
      // More flexible rating check - using getByTestId and looking for partial content
      const bookRating = screen.getByTestId('book-rating');
      expect(bookRating).toBeInTheDocument();
      expect(bookRating.textContent).toContain(mockBook.average_rating.toFixed(1));
      expect(bookRating.textContent).toContain('1,000');
      expect(bookRating.textContent).toContain('ratings');
      
      expect(screen.getByText(`$${mockBook.price}`)).toBeInTheDocument();
      expect(screen.getByText(mockBook.main_category)).toBeInTheDocument();
    });

    it('renders free badge for books with price 0.0', async () => {
      const freeBook = { ...mockBook, price: '0.0' };
      (getBookDetails as jest.Mock).mockResolvedValue(freeBook);

      render(await BookPage({ params: { id: mockBook.parent_asin } }));

      expect(screen.getByText('Free')).toBeInTheDocument();
    });

    it('handles server error gracefully', async () => {
      // Simulate a server error (500)
      (getBookDetails as jest.Mock).mockRejectedValue(new Error('Internal Server Error'));

      const component = await BookPage({ params: { id: mockBook.parent_asin } });
      expect(component).toBeNull();
      expect(notFound).toHaveBeenCalled();
    });

    it('handles network failure gracefully', async () => {
      // Simulate a network failure
      (getBookDetails as jest.Mock).mockRejectedValue(new Error('Network Error'));

      const component = await BookPage({ params: { id: mockBook.parent_asin } });
      expect(component).toBeNull();
      expect(notFound).toHaveBeenCalled();
    });

    it('renders book cover with correct props', async () => {
      render(await BookPage({ params: { id: mockBook.parent_asin } }));

      // Check if book cover is rendered with correct test id
      const bookCover = screen.getByTestId('book-cover');
      expect(bookCover).toBeInTheDocument();
    });
  });

  describe('Metadata Generation', () => {
    it('generates correct metadata for book with subtitle', async () => {
      const metadata = await generateMetadata({ params: { id: mockBook.parent_asin } });

      expect(metadata).toEqual({
        title: mockBook.title,
        description: mockBook.subtitle,
      });
    });

    it('generates correct metadata for book without subtitle', async () => {
      const bookWithoutSubtitle = { ...mockBook, subtitle: null };
      (getBookDetails as jest.Mock).mockResolvedValue(bookWithoutSubtitle);

      const metadata = await generateMetadata({ params: { id: mockBook.parent_asin } });

      expect(metadata).toEqual({
        title: bookWithoutSubtitle.title,
        description: `Details for ${bookWithoutSubtitle.title}`,
      });
    });

    it('generates fallback metadata when book fetch fails', async () => {
      (getBookDetails as jest.Mock).mockRejectedValue(new Error('Failed to fetch'));

      const metadata = await generateMetadata({ params: { id: mockBook.parent_asin } });

      expect(metadata).toEqual({
        title: 'Book Not Found',
        description: 'The requested book could not be found',
      });
    });
  });
}); 