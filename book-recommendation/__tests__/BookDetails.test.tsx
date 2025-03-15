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
    description: 'A great book for testing',
    author: 'Test Author',
    isbn: '1234567890',
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
      
      const encodedData = encodeURIComponent(JSON.stringify(mockBook));
      expect(mockRouter.push).toHaveBeenCalledWith(`/book-details?data=${encodedData}`);
    });
  });

  describe('BookDetails Component', () => {
    it('renders all book details correctly', () => {
      render(<BookDetails data={mockBook} />);

      // Check if all book information is displayed
      expect(screen.getByText('Test Book')).toBeInTheDocument();
      expect(screen.getByText('by Test Author')).toBeInTheDocument();
      expect(screen.getByText('A Test Book')).toBeInTheDocument();
      
      // Check rating using a more flexible matcher
      const ratingElement = screen.getByText(/Rating:/);
      expect(ratingElement).toBeInTheDocument();
      expect(ratingElement.parentElement).toHaveTextContent('4.5');
      expect(ratingElement.parentElement).toHaveTextContent('100 reviews');
      
      // Check price using a more flexible matcher
      const priceElement = screen.getByText(/Price:/);
      expect(priceElement).toBeInTheDocument();
      expect(priceElement.parentElement).toHaveTextContent('$9.99');
      
      expect(screen.getByText('A great book for testing')).toBeInTheDocument();
      
      // Check ISBN using a more flexible matcher
      const isbnElement = screen.getByText(/ISBN:/);
      expect(isbnElement).toBeInTheDocument();
      expect(isbnElement.parentElement).toHaveTextContent('1234567890');
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
        subtitle: 'A Test Book',
        average_rating: 4.5,
        rating_number: 100,
        price: '$9.99',
      };

      render(<BookDetails data={minimalBook} />);

      // Required fields should be present
      expect(screen.getByText('Test Book')).toBeInTheDocument();
      expect(screen.getByText('A Test Book')).toBeInTheDocument();
      
      // Check rating using a more flexible matcher
      const ratingElement = screen.getByText(/Rating:/);
      expect(ratingElement).toBeInTheDocument();
      expect(ratingElement.parentElement).toHaveTextContent('4.5');
      expect(ratingElement.parentElement).toHaveTextContent('100 reviews');
      
      // Check price using a more flexible matcher
      const priceElement = screen.getByText(/Price:/);
      expect(priceElement).toBeInTheDocument();
      expect(priceElement.parentElement).toHaveTextContent('$9.99');

      // Optional fields should not be present
      expect(screen.queryByText(/by/)).not.toBeInTheDocument();
      expect(screen.queryByText('Description')).not.toBeInTheDocument();
      expect(screen.queryByText(/ISBN:/)).not.toBeInTheDocument();
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
      expect(screen.getByText('by Test Author')).toBeInTheDocument();
    });
  });
}); 