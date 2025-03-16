import React from 'react';
import { render, screen } from '@testing-library/react';
import { RecommendationCard } from '../RecommendationCard';
import '@testing-library/jest-dom';
import type { Recommendation } from '@/types/api';

// Mock the necessary components
jest.mock('@/components/book-covers', () => ({
  BookCover: () => <div data-testid="book-cover">Book Cover</div>,
}));

jest.mock('@/components/ui/book-rating', () => ({
  BookRating: ({ 
    rating, 
    ratingCount, 
    size, 
    className 
  }: { 
    rating: number; 
    ratingCount: number; 
    size?: string; 
    className?: string;
  }) => (
    <div 
      data-testid="book-rating"
      data-rating={rating}
      data-rating-count={ratingCount}
      data-size={size}
      className={className}
    >
      Rating: {rating} ({ratingCount})
    </div>
  ),
}));

jest.mock('@/components/ui/book-price', () => ({
  BookPrice: ({ price }: { price: string | number | null }) => (
    <div data-testid="book-price" data-price={price}>
      Price: {price}
    </div>
  ),
}));

jest.mock('@/lib/config', () => ({
  getConfig: jest.fn().mockReturnValue('default'),
}));

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children }: { href: string; children: React.ReactNode }) => (
    <a href={href} data-testid="next-link">
      {children}
    </a>
  ),
}));

describe('RecommendationCard', () => {
  const mockRecommendation: Recommendation = {
    parent_asin: 'B001234567',
    title: 'Test Book',
    subtitle: 'A Test Book for Testing',
    average_rating: 4.5,
    rating_number: 1000,
    price: '9.99',
    score: 0.85,
    image_url: 'http://example.com/image.jpg',
    main_category: 'Fiction'
  };

  it('should render the recommendation card correctly', () => {
    render(
      <RecommendationCard 
        recommendation={mockRecommendation}
      />
    );
    
    // Check that the title and subtitle are displayed
    expect(screen.getByText('Test Book')).toBeInTheDocument();
    expect(screen.getByText('A Test Book for Testing')).toBeInTheDocument();
    
    // Check that the book cover is rendered
    expect(screen.getByTestId('book-cover')).toBeInTheDocument();
    
    // Check that the link is correct
    expect(screen.getByTestId('next-link')).toHaveAttribute('href', '/books/B001234567');
  });

  it('should pass correct rating props to BookRating component', () => {
    render(
      <RecommendationCard 
        recommendation={mockRecommendation}
      />
    );
    
    const ratingElement = screen.getByTestId('book-rating');
    expect(ratingElement).toHaveAttribute('data-rating', '4.5');
    expect(ratingElement).toHaveAttribute('data-rating-count', '1000');
    expect(ratingElement).toHaveAttribute('data-size', 'sm');
    
    // Check that the BookRating has flex-shrink and w-full classes to prevent overflow
    expect(ratingElement).toHaveAttribute('class', 'flex-shrink-0 w-full');
  });

  it('should render price correctly', () => {
    render(
      <RecommendationCard 
        recommendation={mockRecommendation}
      />
    );
    
    const priceElement = screen.getByTestId('book-price');
    expect(priceElement).toHaveAttribute('data-price', '9.99');
  });
  
  it('should show score badge when showScore is true', () => {
    render(
      <RecommendationCard 
        recommendation={mockRecommendation}
        showScore={true}
      />
    );
    
    expect(screen.getByText('Score: 0.85')).toBeInTheDocument();
  });
  
  it('should hide score badge when showScore is false', () => {
    render(
      <RecommendationCard 
        recommendation={mockRecommendation}
        showScore={false}
      />
    );
    
    expect(screen.queryByText('Score: 0.85')).not.toBeInTheDocument();
  });

  it('should handle large rating counts without overflow', () => {
    const recommendationWithLargeCount = {
      ...mockRecommendation,
      rating_number: 9876543,
    };
    
    render(
      <RecommendationCard 
        recommendation={recommendationWithLargeCount}
      />
    );
    
    const ratingElement = screen.getByTestId('book-rating');
    expect(ratingElement).toHaveAttribute('data-rating-count', '9876543');
  });
  
  it('should use custom linkPrefix if provided', () => {
    render(
      <RecommendationCard 
        recommendation={mockRecommendation}
        linkPrefix="/custom"
      />
    );
    
    expect(screen.getByTestId('next-link')).toHaveAttribute('href', '/custom/B001234567');
  });
}); 