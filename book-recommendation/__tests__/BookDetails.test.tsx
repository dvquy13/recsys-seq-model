import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { useRouter, useSearchParams } from 'next/navigation';
import BookDetails from '@/components/BookDetails';
import BookDetailsPage from '@/app/book-details/page';
import RecommendationCard from '@/components/RecommendationCard';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
  useSearchParams: jest.fn(),
}));

describe('Book Details Flow', () => {
  const mockRouter = {
    back: jest.fn(),
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

  describe('RecommendationCard Navigation', () => {
    it('navigates to book details page when clicked', () => {
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
  });

  describe('BookDetails Component', () => {
    it('renders all book details correctly', () => {
      render(<BookDetails data={mockBook} />);

      expect(screen.getByText('Test Book')).toBeInTheDocument();
      expect(screen.getByText('A Test Book')).toBeInTheDocument();
      
      // Check rating
      const ratingText = screen.getByText(/4\.5/);
      expect(ratingText).toBeInTheDocument();
      expect(screen.getByText(/100/)).toBeInTheDocument();
      expect(screen.getByText(/reviews/)).toBeInTheDocument();
      
      // Check price
      expect(screen.getByText(/\$9\.99/)).toBeInTheDocument();
      
      // Check category
      expect(screen.getByText('Fiction')).toBeInTheDocument();
      
      // Check ASIN
      expect(screen.getByText('B123456789')).toBeInTheDocument();
    });

    it('navigates back when back button is clicked', () => {
      render(<BookDetails data={mockBook} />);
      
      const backButton = screen.getByText('Back to Recommendations');
      fireEvent.click(backButton);
      
      expect(mockRouter.back).toHaveBeenCalled();
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

      render(<BookDetails data={minimalBook} />);

      // Required fields should be present
      expect(screen.getByText('Test Book')).toBeInTheDocument();
      
      // Check rating using a more flexible matcher
      const ratingElement = screen.getByText(/Rating:/);
      expect(ratingElement).toBeInTheDocument();
      expect(ratingElement.parentElement).toHaveTextContent('4.5');
      expect(ratingElement.parentElement).toHaveTextContent('100 reviews');
      
      // Check price using a more flexible matcher
      const priceElement = screen.getByText(/Price:/);
      expect(priceElement).toBeInTheDocument();
      expect(priceElement.parentElement).toHaveTextContent('$9.99');

      // Check that empty fields are still rendered but without content
      expect(screen.getByText(/Category:/).parentElement).toHaveTextContent('Category:');
      expect(screen.getByText(/ASIN:/).parentElement).toHaveTextContent('ASIN:');
    });
  });

  describe('BookDetailsPage', () => {
    it('displays error when no data is provided', () => {
      (useSearchParams as jest.Mock).mockReturnValue({
        get: () => null,
      });

      render(<BookDetailsPage />);
      
      expect(screen.getByText('No book data provided')).toBeInTheDocument();
    });

    it('displays error when data is invalid', () => {
      (useSearchParams as jest.Mock).mockReturnValue({
        get: () => 'invalid-json',
      });

      render(<BookDetailsPage />);
      
      expect(screen.getByText('Error parsing book data')).toBeInTheDocument();
    });

    it('renders book details when valid data is provided', () => {
      const encodedData = encodeURIComponent(JSON.stringify(mockBook));
      (useSearchParams as jest.Mock).mockReturnValue({
        get: () => encodedData,
      });

      render(<BookDetailsPage />);
      
      expect(screen.getByText('Test Book')).toBeInTheDocument();
      expect(screen.getByText('A Test Book')).toBeInTheDocument();
      
      // Check rating
      const ratingText = screen.getByText(/4\.5/);
      expect(ratingText).toBeInTheDocument();
      expect(screen.getByText(/100/)).toBeInTheDocument();
      expect(screen.getByText(/reviews/)).toBeInTheDocument();
      
      // Check price
      expect(screen.getByText(/\$9\.99/)).toBeInTheDocument();
      
      // Check category
      expect(screen.getByText('Fiction')).toBeInTheDocument();
      
      // Check ASIN
      expect(screen.getByText('B123456789')).toBeInTheDocument();
    });
  });
}); 