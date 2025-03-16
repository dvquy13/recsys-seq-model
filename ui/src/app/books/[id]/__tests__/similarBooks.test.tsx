import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { SimilarBooksSection } from '@/components/SimilarBooksSection';
import { recommendationsApi } from '@/lib/api';
import '@testing-library/jest-dom';
import type { Recommendation } from '@/types/api';

// Mock the API with proper implementation
jest.mock('@/lib/api', () => ({
  recommendationsApi: {
    getRecommendations: jest.fn(),
  },
}));

// Mock the lucide-react icons
jest.mock('lucide-react', () => ({
  Star: () => <span data-testid="star-icon">★</span>,
  StarHalf: () => <span data-testid="star-half-icon">★</span>,
}));

// Mock the RecommendationsGrid component
jest.mock('@/components/RecommendationsGrid', () => ({
  RecommendationsGrid: ({ 
    title, 
    recommendations 
  }: { 
    title: string; 
    recommendations: Recommendation[] 
  }) => (
    <div data-testid="recommendations-grid">
      <h3 data-testid="recommendations-title">{title}</h3>
      <div data-testid="recommendations-count">{recommendations.length}</div>
      <ul>
        {recommendations.map((rec: Recommendation, index: number) => (
          <li key={index} data-testid={`recommendation-${index}`}>
            {rec.title}
          </li>
        ))}
      </ul>
    </div>
  ),
}));

// Mock localStorage for tests
const mockLocalStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => {
      return store[key] || null;
    }),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value.toString();
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

// Mock personalizedRecs to prevent errors
jest.mock('@/lib/personalizedRecs', () => ({
  getRecommendations: jest.fn().mockResolvedValue({ recommendations: [] }),
  updatePersonalizedRecs: jest.fn().mockResolvedValue(true),
  getCachedPersonalizedRecs: jest.fn().mockReturnValue(null),
  hasPersonalizedRecs: jest.fn().mockReturnValue(false),
  clearPersonalizedRecs: jest.fn(),
  PERSONALIZED_RECS_UPDATED_EVENT: 'personalized-recs-updated',
}));

describe('SimilarBooksSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();
  });

  const mockRecommendations: Recommendation[] = [
    {
      parent_asin: 'B00123ABC1',
      title: 'Book 1',
      subtitle: 'Subtitle 1',
      average_rating: 4.5,
      rating_number: 1000,
      price: '9.99',
      score: 0.85,
      image_url: 'http://example.com/image1.jpg',
      main_category: 'Fiction'
    },
    {
      parent_asin: 'B00123ABC2',
      title: 'Book 2',
      subtitle: 'Subtitle 2',
      average_rating: 4.2,
      rating_number: 2000,
      price: '11.99',
      score: 0.75,
      image_url: 'http://example.com/image2.jpg',
      main_category: 'Fiction'
    }
  ];

  it('should render loading state initially', () => {
    (recommendationsApi.getRecommendations as jest.Mock).mockResolvedValue({
      recommendations: [],
      ctx: { user_ids_raw: [], item_seq_raw: [[]], candidate_items_raw: [] },
      metadata: { rec_id: '123' }
    });

    render(<SimilarBooksSection bookId="B001234ABC" />);
    
    expect(screen.getByText('Loading similar books...')).toBeInTheDocument();
  });

  it('should render recommendations when loaded successfully', async () => {
    (recommendationsApi.getRecommendations as jest.Mock).mockResolvedValue({
      recommendations: mockRecommendations,
      ctx: { user_ids_raw: [], item_seq_raw: [[]], candidate_items_raw: [] },
      metadata: { rec_id: '123' }
    });

    render(<SimilarBooksSection bookId="B001234ABC" />);
    
    // Wait for recommendations to load
    await waitFor(() => {
      expect(screen.getByTestId('recommendations-grid')).toBeInTheDocument();
    });
    
    expect(screen.getByTestId('recommendations-title')).toHaveTextContent('Similar Books You Might Like');
    expect(screen.getByTestId('recommendations-count')).toHaveTextContent('2');
    expect(screen.getByTestId('recommendation-0')).toHaveTextContent('Book 1');
    expect(screen.getByTestId('recommendation-1')).toHaveTextContent('Book 2');
  });

  it('should handle API errors', async () => {
    (recommendationsApi.getRecommendations as jest.Mock).mockRejectedValue(
      new Error('API error')
    );

    render(<SimilarBooksSection bookId="B001234ABC" />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to load recommendations')).toBeInTheDocument();
    });
  });

  it('should not render anything if no recommendations are returned', async () => {
    (recommendationsApi.getRecommendations as jest.Mock).mockResolvedValue({
      recommendations: [],
      ctx: { user_ids_raw: [], item_seq_raw: [[]], candidate_items_raw: [] },
      metadata: { rec_id: '123' }
    });

    render(<SimilarBooksSection bookId="B001234ABC" />);
    
    // Wait for loading to finish
    await waitFor(() => {
      expect(screen.queryByText('Loading similar books...')).not.toBeInTheDocument();
    });
    
    // Should not render the recommendations grid
    expect(screen.queryByTestId('recommendations-grid')).not.toBeInTheDocument();
  });

  it('should call the API with the correct parameters', async () => {
    (recommendationsApi.getRecommendations as jest.Mock).mockResolvedValue({
      recommendations: mockRecommendations,
      ctx: { user_ids_raw: [], item_seq_raw: [[]], candidate_items_raw: [] },
      metadata: { rec_id: '123' }
    });

    render(<SimilarBooksSection bookId="B001234ABC" />);
    
    await waitFor(() => {
      expect(recommendationsApi.getRecommendations).toHaveBeenCalledWith(
        "anonymous", 
        10, 
        false,
        [["B001234ABC"]]
      );
    });
  });

  it('should not make API call if bookId is empty', async () => {
    render(<SimilarBooksSection bookId="" />);
    
    await waitFor(() => {
      expect(recommendationsApi.getRecommendations).not.toHaveBeenCalled();
    });
    
    expect(screen.queryByText('Loading similar books...')).not.toBeInTheDocument();
  });
}); 