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

  const mockBook = {
    image_url: 'https://example.com/book.jpg',
    title: 'Test Book',
    subtitle: 'A Test Book',
    average_rating: 4.5,
    rating_number: 100,
    price: '$9.99',
    score: 0.95,
    main_category: 'Fiction',
    parent_asin: 'B123456789'
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (useRouter as jest.Mock).mockReturnValue(mockRouter);
  });

  it('renders recommendation card with all details', () => {
    render(<RecommendationCard recommendation={mockBook} />);

    // Find by heading for unique title match
    expect(screen.getByRole('heading', { name: 'Test Book' })).toBeInTheDocument();
    expect(screen.getByText(/Rating: 4.5 \(100 reviews\)/)).toBeInTheDocument();
    expect(screen.getByText('Price: $$9.99')).toBeInTheDocument();
    expect(screen.getByAltText('Test Book')).toBeInTheDocument();
  });

  it('navigates to book details page with correct data when clicked', () => {
    render(<RecommendationCard recommendation={mockBook} />);
    
    const card = screen.getByText('Test Book').closest('div');
    expect(card).toBeInTheDocument();
    
    fireEvent.click(card!);
    
    // Get the actual URL that was called
    const actualUrl = mockRouter.push.mock.calls[0][0];
    expect(actualUrl).toMatch(/^\/book-details\?data=/);
    
    // Extract and parse the data parameter
    const dataParam = decodeURIComponent(actualUrl.split('data=')[1]);
    const actualData = JSON.parse(dataParam);
    
    // Compare the actual data with expected data
    expect(actualData).toEqual(mockBook);
  });

  it('uses BookCover component for placeholder images', () => {
    const mockBookWithPlaceholder = {
      ...mockBook,
      image_url: 'https://placehold.co/400x600'
    };

    render(<RecommendationCard recommendation={mockBookWithPlaceholder} />);
    
    // Verify BookCover is rendered
    expect(screen.getByTestId('book-cover')).toBeInTheDocument();
    
    // Click the card
    const card = screen.getAllByText('Test Book')[0].closest('div');
    fireEvent.click(card!);
    
    // Get the actual URL that was called
    const actualUrl = mockRouter.push.mock.calls[0][0];
    expect(actualUrl).toMatch(/^\/book-details\?data=/);
    
    // Extract and parse the data parameter
    const dataParam = decodeURIComponent(actualUrl.split('data=')[1]);
    const actualData = JSON.parse(dataParam);
    
    // Compare the actual data with expected data
    expect(actualData).toEqual({
      ...mockBook,
      image_url: ''
    });
  });

  it('handles missing optional fields gracefully', () => {
    const minimalBook = {
      image_url: 'https://example.com/book.jpg',
      title: 'Test Book',
      subtitle: '',
      average_rating: 4.5,
      rating_number: 100,
      price: '$9.99',
      score: 0,
      main_category: '',
      parent_asin: ''
    };

    render(<RecommendationCard recommendation={minimalBook} />);
    
    const card = screen.getByText('Test Book').closest('div');
    fireEvent.click(card!);
    
    // Get the actual URL that was called
    const actualUrl = mockRouter.push.mock.calls[0][0];
    expect(actualUrl).toMatch(/^\/book-details\?data=/);
    
    // Extract and parse the data parameter
    const dataParam = decodeURIComponent(actualUrl.split('data=')[1]);
    const actualData = JSON.parse(dataParam);
    
    // Compare the actual data with expected data
    expect(actualData).toEqual(minimalBook);
  });

  it('uses fallback data when there is an error', () => {
    const mockError = new Error('Test error');
    jest.spyOn(JSON, 'stringify').mockImplementationOnce(() => {
      throw mockError;
    });

    render(<RecommendationCard recommendation={mockBook} />);
    
    const card = screen.getByText('Test Book').closest('div');
    fireEvent.click(card!);
    
    // Get the actual URL that was called
    const actualUrl = mockRouter.push.mock.calls[0][0];
    expect(actualUrl).toMatch(/^\/book-details\?data=/);
    
    // Extract and parse the data parameter
    const dataParam = decodeURIComponent(actualUrl.split('data=')[1]);
    const actualData = JSON.parse(dataParam);
    
    // Compare the actual data with expected fallback data
    expect(actualData).toEqual({
      image_url: '',
      title: 'Test Book',
      subtitle: 'A Test Book',
      average_rating: 0,
      rating_number: 0,
      price: '0',
      main_category: '',
      score: 0,
      parent_asin: ''
    });
  });
}); 