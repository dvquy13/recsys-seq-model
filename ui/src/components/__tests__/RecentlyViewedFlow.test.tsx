import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BookViewTracker } from '../BookViewTracker';
import { RecentlyViewedGrid } from '../RecentlyViewedGrid';
import { addRecentlyViewedBook, getRecentlyViewedBooks } from '@/lib/recentlyViewed';
import { updatePersonalizedRecs } from '@/lib/personalizedRecs';
import type { Recommendation } from '@/types/api';
import { MAX_DISPLAY_ITEMS } from '@/lib/config/recentlyViewed';

// Mock the dependencies
jest.mock('@/lib/recentlyViewed', () => ({
  addRecentlyViewedBook: jest.fn(),
  getRecentlyViewedBooks: jest.fn(),
  RECENTLY_VIEWED_CHANGE_EVENT: 'recently-viewed-change'
}));

jest.mock('@/lib/personalizedRecs', () => ({
  updatePersonalizedRecs: jest.fn().mockImplementation(() => Promise.resolve(true))
}));

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    })
  };
})();

Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// Mock next/image
jest.mock('next/image', () => ({
  __esModule: true,
  default: ({ src, alt }: { src: string; alt: string }) => (
    <img src={src} alt={alt} />
  ),
}));

describe('Recently Viewed Flow', () => {
  const mockBook: Recommendation = {
    parent_asin: 'B001BOOK1',
    title: 'Test Book 1',
    subtitle: 'A Test Book',
    main_category: 'Fiction',
    average_rating: 4.5,
    rating_number: 100,
    price: '9.99',
    image_url: 'https://example.com/book1.jpg',
    score: 0.95
  };

  const mockBook2: Recommendation = {
    ...mockBook,
    parent_asin: 'B002BOOK2',
    title: 'Test Book 2'
  };

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue([]);
  });

  it('should track book view and update recently viewed list', async () => {
    // Render BookViewTracker with a book
    await act(async () => {
      render(<BookViewTracker book={mockBook} />);
    });

    // Verify that addRecentlyViewedBook was called with the book
    expect(addRecentlyViewedBook).toHaveBeenCalledWith(expect.objectContaining({
      parent_asin: mockBook.parent_asin,
      title: mockBook.title
    }));

    // Verify that updatePersonalizedRecs was called
    expect(updatePersonalizedRecs).toHaveBeenCalled();
  });

  it('should update recently viewed grid when new book is viewed', async () => {
    // Mock getRecentlyViewedBooks to return our test books
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue([
      { ...mockBook, viewedAt: Date.now() }
    ]);

    // Render the RecentlyViewedGrid
    await act(async () => {
      render(<RecentlyViewedGrid />);
    });

    // Verify the first book is displayed
    expect(screen.getByText(mockBook.title)).toBeInTheDocument();

    // Simulate viewing a second book
    await act(async () => {
      render(<BookViewTracker book={mockBook2} />);
    });

    // Update the mock to include both books
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue([
      { ...mockBook2, viewedAt: Date.now() + 1000 },
      { ...mockBook, viewedAt: Date.now() }
    ]);

    // Dispatch the recently viewed change event
    await act(async () => {
      window.dispatchEvent(new Event('recently-viewed-change'));
    });

    // Verify both books are now displayed in the correct order
    expect(screen.getByText(mockBook2.title)).toBeInTheDocument();
    expect(screen.getByText(mockBook.title)).toBeInTheDocument();
  });

  it('should handle guest users (no user ID)', async () => {
    // Clear any existing user ID
    localStorageMock.removeItem('last-submitted-user-id');

    // Render BookViewTracker with a book
    await act(async () => {
      render(<BookViewTracker book={mockBook} />);
    });

    // Verify that updatePersonalizedRecs was called with 'guest'
    expect(updatePersonalizedRecs).toHaveBeenCalledWith('guest');

    // Verify the book was still added to recently viewed
    expect(addRecentlyViewedBook).toHaveBeenCalledWith(expect.objectContaining({
      parent_asin: mockBook.parent_asin
    }));
  });

  it('should handle logged-in users', async () => {
    // Set a user ID
    const testUserId = 'test-user-123';
    localStorageMock.setItem('last-submitted-user-id', testUserId);

    // Render BookViewTracker with a book
    await act(async () => {
      render(<BookViewTracker book={mockBook} />);
    });

    // Verify that updatePersonalizedRecs was called with the user ID
    expect(updatePersonalizedRecs).toHaveBeenCalledWith(testUserId);
  });

  it('should limit recently viewed books display based on MAX_DISPLAY_ITEMS', async () => {
    // Create more books than the display limit
    const totalBooks = MAX_DISPLAY_ITEMS + 2; // Create 2 extra books beyond the limit
    const books = Array.from({ length: totalBooks }, (_, i) => ({
      ...mockBook,
      parent_asin: `B00${i}BOOK`,
      title: `Test Book ${i + 1}`,
      viewedAt: Date.now() + i * 1000 // Most recent books have higher timestamps
    }));

    // Mock getRecentlyViewedBooks to return all books
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue(books);

    // Render the RecentlyViewedGrid
    await act(async () => {
      render(<RecentlyViewedGrid />);
    });

    // Verify the expected number of books are displayed
    // Books should be displayed in reverse order (newest first)
    for (let i = 0; i < MAX_DISPLAY_ITEMS; i++) {
      const bookIndex = totalBooks - i; // Start from the newest book
      expect(screen.getByText(`Test Book ${bookIndex}`)).toBeInTheDocument();
    }

    // Verify books beyond the limit are not displayed
    for (let i = 1; i <= totalBooks - MAX_DISPLAY_ITEMS; i++) {
      expect(screen.queryByText(`Test Book ${i}`)).not.toBeInTheDocument();
    }
  });
}); 