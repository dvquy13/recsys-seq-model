import React from 'react';
import { render } from '@testing-library/react';
import { BookViewTracker } from '../BookViewTracker';
import { addRecentlyViewedBook } from '@/lib/recentlyViewed';

// Mock dependencies
jest.mock('@/lib/recentlyViewed', () => ({
  addRecentlyViewedBook: jest.fn()
}));

describe('BookViewTracker', () => {
  const mockBook = {
    parent_asin: '123',
    title: 'Test Book',
    subtitle: 'A Test Subtitle',
    main_category: 'Fiction',
    average_rating: 4.5,
    rating_number: 100,
    price: '9.99',
    image_url: 'https://example.com/image.jpg',
    score: 0.95
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should call addRecentlyViewedBook with the book when mounted', () => {
    render(<BookViewTracker book={mockBook} />);
    
    expect(addRecentlyViewedBook).toHaveBeenCalledTimes(1);
    expect(addRecentlyViewedBook).toHaveBeenCalledWith(expect.objectContaining({
      parent_asin: mockBook.parent_asin,
      title: mockBook.title
    }));
  });

  it('should not render any DOM elements', () => {
    const { container } = render(<BookViewTracker book={mockBook} />);
    expect(container.firstChild).toBeNull();
  });

  it('should validate book properties before adding to recently viewed', () => {
    // Book with missing properties
    const incompleteBook = {
      parent_asin: '123',
      // Missing title
    } as any;
    
    render(<BookViewTracker book={incompleteBook} />);
    
    // Should not add book without required properties
    expect(addRecentlyViewedBook).not.toHaveBeenCalled();
  });

  it('should sanitize book data before adding to recently viewed', () => {
    // Book with some missing optional properties
    const partialBook = {
      parent_asin: '123',
      title: 'Test Book',
      // Missing other fields
    } as any;
    
    render(<BookViewTracker book={partialBook} />);
    
    // Should add book with sanitized data
    expect(addRecentlyViewedBook).toHaveBeenCalledTimes(1);
    expect(addRecentlyViewedBook).toHaveBeenCalledWith(expect.objectContaining({
      parent_asin: '123',
      title: 'Test Book',
      subtitle: '',
      main_category: 'Unknown',
      average_rating: 0,
      rating_number: 0,
      price: null,
      image_url: '',
      score: 0
    }));
  });
}); 