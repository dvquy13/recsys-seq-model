import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { useRouter } from 'next/navigation';
import RecommendationCard from '@/components/RecommendationCard';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

describe('RecommendationCard', () => {
  const mockRouter = {
    push: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (useRouter as jest.Mock).mockReturnValue(mockRouter);
  });

  const mockRecommendation = {
    image_url: 'https://example.com/book.jpg',
    title: 'Test Book',
    subtitle: 'A Test Book',
    average_rating: 4.5,
    rating_number: 100,
    price: '$9.99',
    description: 'A great book for testing',
    author: 'Test Author',
    isbn: '1234567890',
  };

  it('renders recommendation card with all details', () => {
    render(<RecommendationCard recommendation={mockRecommendation} />);

    // Find by heading for unique title match
    expect(screen.getByRole('heading', { name: 'Test Book' })).toBeInTheDocument();
    expect(screen.getByText('A Test Book')).toBeInTheDocument();
    expect(screen.getByText(/Rating: 4.5 \(100 reviews\)/)).toBeInTheDocument();
    expect(screen.getByText('Price: $9.99')).toBeInTheDocument();
    expect(screen.getByAltText('Test Book')).toBeInTheDocument();
  });

  it('navigates to book details page with correct data when clicked', () => {
    render(<RecommendationCard recommendation={mockRecommendation} />);
    
    // Find the clickable container
    const card = screen.getByRole('heading', { name: 'Test Book' }).closest('div[style*="cursor: pointer"]');
    expect(card).toBeInTheDocument();
    
    fireEvent.click(card!);

    const expectedData = {
      ...mockRecommendation,
      average_rating: Number(mockRecommendation.average_rating),
      rating_number: Number(mockRecommendation.rating_number),
    };
    
    const encodedData = encodeURIComponent(JSON.stringify(expectedData));
    expect(mockRouter.push).toHaveBeenCalledWith(`/book-details?data=${encodedData}`);
  });

  it('uses BookCover component for placeholder images', () => {
    const placeholderRecommendation = {
      ...mockRecommendation,
      image_url: 'https://placehold.co/book.jpg',
    };

    render(<RecommendationCard recommendation={placeholderRecommendation} />);
    
    // BookCover should be rendered instead of Image
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    
    // Click should pass empty image_url to trigger BookCover in details view
    const card = screen.getByRole('heading', { name: 'Test Book' }).closest('div[style*="cursor: pointer"]');
    fireEvent.click(card!);

    const expectedData = {
      ...placeholderRecommendation,
      image_url: '', // Empty URL triggers BookCover
      average_rating: Number(placeholderRecommendation.average_rating),
      rating_number: Number(placeholderRecommendation.rating_number),
    };
    
    const encodedData = encodeURIComponent(JSON.stringify(expectedData));
    expect(mockRouter.push).toHaveBeenCalledWith(`/book-details?data=${encodedData}`);
  });

  it('handles missing optional fields gracefully', () => {
    const minimalRecommendation = {
      image_url: 'https://example.com/book.jpg',
      title: 'Test Book',
      subtitle: 'A Test Book',
      average_rating: 4.5,
      rating_number: 100,
      price: '$9.99',
    };

    render(<RecommendationCard recommendation={minimalRecommendation} />);
    
    expect(screen.getByRole('heading', { name: 'Test Book' })).toBeInTheDocument();
    expect(screen.getByText('A Test Book')).toBeInTheDocument();
    expect(screen.getByText(/Rating: 4.5 \(100 reviews\)/)).toBeInTheDocument();
    expect(screen.getByText('Price: $9.99')).toBeInTheDocument();

    const card = screen.getByRole('heading', { name: 'Test Book' }).closest('div[style*="cursor: pointer"]');
    fireEvent.click(card!);

    const expectedData = {
      ...minimalRecommendation,
      average_rating: Number(minimalRecommendation.average_rating),
      rating_number: Number(minimalRecommendation.rating_number),
    };
    
    const encodedData = encodeURIComponent(JSON.stringify(expectedData));
    expect(mockRouter.push).toHaveBeenCalledWith(`/book-details?data=${encodedData}`);
  });

  it('uses fallback data when there is an error', () => {
    // Mock JSON.stringify to throw an error
    const originalStringify = JSON.stringify;
    JSON.stringify = jest.fn().mockImplementationOnce(() => {
      throw new Error('Circular reference');
    });

    render(<RecommendationCard recommendation={mockRecommendation} />);
    
    const card = screen.getByRole('heading', { name: 'Test Book' }).closest('div[style*="cursor: pointer"]');
    fireEvent.click(card!);

    const expectedFallback = {
      image_url: '',
      title: 'Test Book',
      subtitle: '',
      average_rating: 0,
      rating_number: 0,
      price: '0',
    };
    
    const encodedFallback = encodeURIComponent(JSON.stringify(expectedFallback));
    expect(mockRouter.push).toHaveBeenCalledWith(`/book-details?data=${encodedFallback}`);

    // Restore original JSON.stringify
    JSON.stringify = originalStringify;
  });
}); 