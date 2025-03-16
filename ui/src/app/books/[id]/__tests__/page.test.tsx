import React from 'react';
import { render, screen } from '@testing-library/react';
import BookPage, { generateMetadata } from '../page';
import { getBookDetails } from '@/lib/api';
import { notFound } from 'next/navigation';

// Mock the next/navigation module
jest.mock('next/navigation', () => ({
  notFound: jest.fn(),
}));

// Mock the api module
jest.mock('@/lib/api', () => ({
  getBookDetails: jest.fn(),
}));

// Mock the config module
jest.mock('@/lib/config', () => ({
  getConfig: jest.fn(() => 'textBased'),
}));

describe('BookPage', () => {
  const mockBook = {
    title: 'Test Book',
    subtitle: 'A Test Subtitle',
    average_rating: 4.5,
    rating_number: 1000,
    price: '9.99',
    main_category: 'Fiction',
    image_url: 'https://example.com/book.jpg',
    parent_asin: 'B000ASPUES',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (getBookDetails as jest.Mock).mockResolvedValue(mockBook);
  });

  describe('Page Component', () => {
    it('renders book details correctly', async () => {
      render(await BookPage({ params: { id: mockBook.parent_asin } }));

      // Check if main book information is rendered
      expect(screen.getByText(mockBook.title)).toBeInTheDocument();
      expect(screen.getByText(mockBook.subtitle)).toBeInTheDocument();
      // Check for both rating value and text
      expect(screen.getByText(mockBook.average_rating.toFixed(1))).toBeInTheDocument();
      expect(screen.getByText(`(${mockBook.rating_number.toLocaleString()} ratings)`)).toBeInTheDocument();
      expect(screen.getByText(`$${mockBook.price}`)).toBeInTheDocument();
      expect(screen.getByText(mockBook.main_category)).toBeInTheDocument();
    });

    it('renders free badge for books with price 0.0', async () => {
      const freeBook = { ...mockBook, price: '0.0' };
      (getBookDetails as jest.Mock).mockResolvedValue(freeBook);

      render(await BookPage({ params: { id: mockBook.parent_asin } }));

      expect(screen.getByText('Free')).toBeInTheDocument();
    });

    it('handles server error gracefully', async () => {
      // Simulate a server error (500)
      (getBookDetails as jest.Mock).mockRejectedValue(new Error('Internal Server Error'));

      const component = await BookPage({ params: { id: mockBook.parent_asin } });
      expect(component).toBeNull();
      expect(notFound).toHaveBeenCalled();
    });

    it('handles network failure gracefully', async () => {
      // Simulate a network failure
      (getBookDetails as jest.Mock).mockRejectedValue(new Error('Network Error'));

      const component = await BookPage({ params: { id: mockBook.parent_asin } });
      expect(component).toBeNull();
      expect(notFound).toHaveBeenCalled();
    });

    it('renders book cover with correct props', async () => {
      render(await BookPage({ params: { id: mockBook.parent_asin } }));

      // Check if book cover is rendered with correct test id
      const bookCover = screen.getByTestId('book-cover');
      expect(bookCover).toBeInTheDocument();
    });
  });

  describe('Metadata Generation', () => {
    it('generates correct metadata for book with subtitle', async () => {
      const metadata = await generateMetadata({ params: { id: mockBook.parent_asin } });

      expect(metadata).toEqual({
        title: mockBook.title,
        description: mockBook.subtitle,
      });
    });

    it('generates correct metadata for book without subtitle', async () => {
      const bookWithoutSubtitle = { ...mockBook, subtitle: null };
      (getBookDetails as jest.Mock).mockResolvedValue(bookWithoutSubtitle);

      const metadata = await generateMetadata({ params: { id: mockBook.parent_asin } });

      expect(metadata).toEqual({
        title: bookWithoutSubtitle.title,
        description: `Details for ${bookWithoutSubtitle.title}`,
      });
    });

    it('generates fallback metadata when book fetch fails', async () => {
      (getBookDetails as jest.Mock).mockRejectedValue(new Error('Failed to fetch'));

      const metadata = await generateMetadata({ params: { id: mockBook.parent_asin } });

      expect(metadata).toEqual({
        title: 'Book Not Found',
        description: 'The requested book could not be found',
      });
    });
  });
}); 