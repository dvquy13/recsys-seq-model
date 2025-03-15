import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import { RecentlyViewedGrid } from '../RecentlyViewedGrid';
import { getRecentlyViewedBooks, RECENTLY_VIEWED_CHANGE_EVENT } from '@/lib/recentlyViewed';
import { RecommendationsGrid } from '../RecommendationsGrid';

// Mock dependencies
jest.mock('@/lib/recentlyViewed', () => ({
  getRecentlyViewedBooks: jest.fn(),
  RECENTLY_VIEWED_CHANGE_EVENT: 'recently-viewed-change'
}));

jest.mock('../RecommendationsGrid', () => ({
  RecommendationsGrid: jest.fn(() => <div data-testid="recommendations-grid" />)
}));

describe('RecentlyViewedGrid', () => {
  const mockBooks = [
    {
      parent_asin: '123',
      title: 'Book 1',
      subtitle: 'Subtitle 1',
      main_category: 'Fiction',
      average_rating: 4.5,
      rating_number: 100,
      price: '9.99',
      image_url: 'https://example.com/image1.jpg',
      score: 0.95,
      viewedAt: 3000
    },
    {
      parent_asin: '456',
      title: 'Book 2',
      subtitle: 'Subtitle 2',
      main_category: 'Non-Fiction',
      average_rating: 4.0,
      rating_number: 200,
      price: '14.99',
      image_url: 'https://example.com/image2.jpg',
      score: 0.85,
      viewedAt: 2000
    },
    {
      parent_asin: '789',
      title: 'Book 3',
      subtitle: 'Subtitle 3',
      main_category: 'Science',
      average_rating: 3.5,
      rating_number: 150,
      price: '19.99',
      image_url: 'https://example.com/image3.jpg',
      score: 0.75,
      viewedAt: 1000
    },
    {
      parent_asin: '101',
      title: 'Book 4',
      subtitle: 'Subtitle 4',
      main_category: 'History',
      average_rating: 3.0,
      rating_number: 120,
      price: '12.99',
      image_url: 'https://example.com/image4.jpg',
      score: 0.65,
      viewedAt: 500
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should not render when there are no recently viewed books', async () => {
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue([]);
    
    await act(async () => {
      render(<RecentlyViewedGrid />);
    });
    
    // RecommendationsGrid should not be called when there are no books
    expect(RecommendationsGrid).not.toHaveBeenCalled();
    
    // The component should not render anything
    expect(screen.queryByTestId('recommendations-grid')).not.toBeInTheDocument();
  });

  it('should render RecommendationsGrid with a maximum of 3 books when books are available', async () => {
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue(mockBooks);
    
    await act(async () => {
      render(<RecentlyViewedGrid />);
    });
    
    expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument();
    
    // Check that RecommendationsGrid was called
    expect(RecommendationsGrid).toHaveBeenCalled();
    
    // Get the recommendations that were passed to RecommendationsGrid
    const passedProps = (RecommendationsGrid as jest.Mock).mock.calls[0][0];
    
    // Check the title prop
    expect(passedProps.title).toBe('Recently Viewed');
    
    // Check the recommendations array
    const passedRecommendations = passedProps.recommendations;
    
    // Verify we have exactly 3 books
    expect(passedRecommendations.length).toBe(3);
    
    // Verify they are the expected books in the correct order
    expect(passedRecommendations[0].parent_asin).toBe('123'); // Most recent (viewedAt: 3000)
    expect(passedRecommendations[1].parent_asin).toBe('456'); // Second (viewedAt: 2000)
    expect(passedRecommendations[2].parent_asin).toBe('789'); // Third (viewedAt: 1000)
    
    // Book 4 should not be included (4th most recent)
    expect(passedRecommendations.find((book: any) => book.parent_asin === '101')).toBeUndefined();
  });

  it('should reload books when the RECENTLY_VIEWED_CHANGE_EVENT is triggered', async () => {
    // Initially return empty array
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue([]);
    
    await act(async () => {
      render(<RecentlyViewedGrid />);
    });
    
    // Initially RecommendationsGrid should not be rendered
    expect(RecommendationsGrid).not.toHaveBeenCalled();
    
    // Now return books and trigger the event
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue(mockBooks);
    
    await act(async () => {
      window.dispatchEvent(new Event(RECENTLY_VIEWED_CHANGE_EVENT));
    });
    
    // Wait for the component to update
    await waitFor(() => {
      expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument();
      expect(RecommendationsGrid).toHaveBeenCalled();
    });
  });

  it('should filter out books with missing required properties', async () => {
    const booksWithInvalidData = [
      ...mockBooks,
      {
        // Missing title
        parent_asin: '999',
        subtitle: 'Invalid Book',
        viewedAt: 4000
      } as any,
      {
        // Missing parent_asin
        title: 'Invalid Book 2',
        viewedAt: 5000
      } as any
    ];
    
    (getRecentlyViewedBooks as jest.Mock).mockReturnValue(booksWithInvalidData);
    
    await act(async () => {
      render(<RecentlyViewedGrid />);
    });
    
    // Should still only pass the valid books
    const passedRecommendations = (RecommendationsGrid as jest.Mock).mock.calls[0][0].recommendations;
    expect(passedRecommendations.length).toBe(3); // Only valid books, max 3
    
    // Make sure invalid books are not included
    expect(passedRecommendations.find((book: any) => book.subtitle === 'Invalid Book')).toBeUndefined();
    expect(passedRecommendations.find((book: any) => book.title === 'Invalid Book 2')).toBeUndefined();
  });
}); 