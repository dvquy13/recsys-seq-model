import React from 'react';
import { render } from '@testing-library/react';
import { BookViewTracker } from '../BookViewTracker';
import * as AppStateContext from '@/providers/app-state-provider';

// Mock the useAppState hook
jest.mock('@/providers/app-state-provider', () => ({
  useAppState: jest.fn()
}));

describe('BookViewTracker', () => {
  // Create mock functions
  const mockAddBookToRecentlyViewed = jest.fn();
  const mockUpdatePersonalizedRecommendations = jest.fn();
  
  // Create a mock book
  const mockBook = {
    parent_asin: '123',
    title: 'Test Book',
    average_rating: 4.5,
    rating_number: 100,
    price: '9.99',
    subtitle: 'Test Subtitle',
    main_category: 'Fiction',
    score: 0.9,
    image_url: 'https://example.com/test.jpg'
  };
  
  beforeEach(() => {
    // Setup the mock implementation for useAppState
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      addBookToRecentlyViewed: mockAddBookToRecentlyViewed,
      updatePersonalizedRecommendations: mockUpdatePersonalizedRecommendations,
      // Add other required properties but with empty/mock values
      userId: '',
      setUserId: jest.fn(),
      submittedUserId: null,
      submitUserId: jest.fn(),
      recentlyViewedBooks: [],
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: false,
      recommendationsError: null
    });
  });
  
  afterEach(() => {
    jest.clearAllMocks();
  });
  
  it('should call addBookToRecentlyViewed with the book when mounted', () => {
    render(<BookViewTracker book={mockBook} />);
    
    expect(mockAddBookToRecentlyViewed).toHaveBeenCalledTimes(1);
    expect(mockAddBookToRecentlyViewed).toHaveBeenCalledWith(expect.objectContaining({
      parent_asin: mockBook.parent_asin,
      title: mockBook.title
    }));
  });
  
  it('should not call addBookToRecentlyViewed if book is missing required properties', () => {
    // Create an invalid book missing title
    const invalidBook = {
      parent_asin: '123',
      average_rating: 4.5,
      rating_number: 100,
      price: 9.99
    };
    
    // @ts-ignore - Deliberately providing incomplete props for testing
    render(<BookViewTracker book={invalidBook} />);
    
    // Should not add book because it's missing the title
    expect(mockAddBookToRecentlyViewed).not.toHaveBeenCalled();
  });
  
  it('should sanitize book data before adding to recently viewed', () => {
    // Create a minimal book with just the required properties
    const minimalBook = {
      parent_asin: '123',
      title: 'Test Book'
    };
    
    // @ts-ignore - Deliberately providing incomplete props for testing
    render(<BookViewTracker book={minimalBook} />);
    
    // Should add book with sanitized data
    expect(mockAddBookToRecentlyViewed).toHaveBeenCalledTimes(1);
    expect(mockAddBookToRecentlyViewed).toHaveBeenCalledWith(expect.objectContaining({
      parent_asin: '123',
      title: 'Test Book',
    }));
  });
}); 