import React from 'react';
import { render, screen } from '@testing-library/react';
import { RecentlyViewedGrid } from '../RecentlyViewedGrid';
import * as AppStateContext from '@/providers/app-state-provider';
import type { Recommendation } from '@/types/api';

// Mock RecommendationsGrid since we're only testing RecentlyViewedGrid
jest.mock('../RecommendationsGrid', () => ({
  RecommendationsGrid: jest.fn(() => <div data-testid="recommendations-grid" />)
}));

// Mock the useAppState hook
jest.mock('@/providers/app-state-provider', () => ({
  useAppState: jest.fn()
}));

describe('RecentlyViewedGrid', () => {
  // Sample books for testing
  const sampleBooks: Recommendation[] = [
    {
      parent_asin: 'book1',
      title: 'Book 1',
      average_rating: 4.5,
      rating_number: 100,
      price: '9.99',
      subtitle: 'Subtitle 1',
      main_category: 'Fiction',
      score: 0.95,
      image_url: 'https://example.com/book1.jpg'
    },
    {
      parent_asin: 'book2',
      title: 'Book 2',
      average_rating: 4.2,
      rating_number: 200,
      price: '8.99',
      subtitle: 'Subtitle 2',
      main_category: 'Non-Fiction',
      score: 0.9,
      image_url: 'https://example.com/book2.jpg'
    },
    {
      parent_asin: 'book3',
      title: 'Book 3',
      average_rating: 4.7,
      rating_number: 150,
      price: '10.99',
      subtitle: 'Subtitle 3',
      main_category: 'Mystery',
      score: 0.92,
      image_url: 'https://example.com/book3.jpg'
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render RecommendationsGrid with recently viewed books', () => {
    // Import the actual RecommendationsGrid mock
    const { RecommendationsGrid } = require('../RecommendationsGrid');
    
    // Setup the mock implementation for useAppState
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      recentlyViewedBooks: sampleBooks,
      // Add other required properties with empty/mock values
      userId: '',
      setUserId: jest.fn(),
      submittedUserId: null,
      submitUserId: jest.fn(),
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: false,
      recommendationsError: null
    });

    render(<RecentlyViewedGrid />);
    
    expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument();
    
    // Check that RecommendationsGrid was called
    expect(RecommendationsGrid).toHaveBeenCalled();
    
    // Get the props passed to RecommendationsGrid
    const props = RecommendationsGrid.mock.calls[0][0];
    
    // Check the props
    expect(props.title).toBe('Recently Viewed');
    expect(props.recommendations).toEqual(sampleBooks);
    expect(props.emptyMessage).toBe('No recently viewed books');
  });

  it('should not render anything when no recently viewed books are available', () => {
    // Setup the mock implementation for useAppState with empty books array
    jest.mocked(AppStateContext.useAppState).mockReturnValue({
      recentlyViewedBooks: [],
      // Add other required properties with empty/mock values
      userId: '',
      setUserId: jest.fn(),
      submittedUserId: null,
      submitUserId: jest.fn(),
      addBookToRecentlyViewed: jest.fn(),
      clearRecentlyViewed: jest.fn(),
      personalizedRecs: null,
      updatePersonalizedRecommendations: jest.fn().mockResolvedValue(true),
      clearPersonalizedRecommendations: jest.fn(),
      recommendations: null,
      isLoadingRecommendations: false,
      recommendationsError: null
    });

    const { container } = render(<RecentlyViewedGrid />);
    
    // Component should not render anything when there are no books
    expect(container.firstChild).toBeNull();
    
    // Import the actual RecommendationsGrid mock
    const { RecommendationsGrid } = require('../RecommendationsGrid');
    
    // RecommendationsGrid should not be called
    expect(RecommendationsGrid).not.toHaveBeenCalled();
  });
}); 