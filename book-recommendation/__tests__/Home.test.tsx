import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Home from '@/app/page';

// Mock Next Image component
jest.mock('next/image', () => ({
  __esModule: true,
  default: (props: any) => {
    // eslint-disable-next-line @next/next/no-img-element
    return <img alt={props.alt} src={props.src} {...props} />;
  },
}));

// Mock useRouter
jest.mock('next/navigation', () => ({
  useRouter() {
    return {
      push: jest.fn(),
      prefetch: jest.fn(),
    };
  },
}));

// Mock fetch globally
global.fetch = jest.fn();

describe('Home Component', () => {
  beforeEach(() => {
    // Clear mock before each test
    (global.fetch as jest.Mock).mockClear();
  });

  it('renders the form with default user ID', () => {
    render(<Home />);
    const input = screen.getByPlaceholderText('Enter User ID') as HTMLInputElement;
    expect(input.value).toBe('AE224PFXAEAT66IXX43GRJSWHXCA');
  });

  it('fetches recommendations and displays them with images', async () => {
    const mockRecommendations = {
      recommendations: [
        {
          id: '1',
          title: 'Test Book',
          author: 'Test Author',
          image_url: 'https://m.media-amazon.com/images/test.jpg',
          average_rating: 4.5,
          rating_number: 100,
          price: 9.99,
          description: 'Test description'
        }
      ]
    };

    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockRecommendations)
      })
    );

    render(<Home />);
    
    const form = screen.getByRole('form');
    fireEvent.submit(form);

    // Check if loading state is shown
    expect(screen.getByText('Loading recommendations...')).toBeInTheDocument();

    // Wait for recommendations to be displayed
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/recs/retrieve?count=10&debug=false',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            accept: 'application/json',
          },
          body: JSON.stringify({
            user_ids_raw: ['AE224PFXAEAT66IXX43GRJSWHXCA'],
            item_seq_raw: [['0439064864', '043935806X']],
            candidate_items_raw: [],
          }),
        })
      );
    });

    // Verify book details and image are displayed
    await waitFor(() => {
      // Find all elements containing the book title
      const titleElements = screen.getAllByText('Test Book');
      expect(titleElements.length).toBeGreaterThan(0);

      // Find the rating text
      const ratingText = screen.getByText('Rating: 4.5 (100 reviews)');
      expect(ratingText).toBeInTheDocument();

      // Check for the image
      const image = screen.getByAltText('Test Book') as HTMLImageElement;
      expect(image).toBeInTheDocument();
      expect(image.src).toBe('https://m.media-amazon.com/images/test.jpg');
    });
  });

  it('handles fetch error gracefully', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      Promise.reject(new Error('API Error'))
    );

    render(<Home />);
    
    const form = screen.getByRole('form');
    fireEvent.submit(form);

    // Check if loading state is shown initially
    expect(screen.getByText('Loading recommendations...')).toBeInTheDocument();

    // Wait for error to be handled
    await waitFor(() => {
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Error fetching recommendations:',
        expect.any(Error)
      );
    });

    consoleErrorSpy.mockRestore();
  });

  it('handles missing or invalid image URLs', async () => {
    const mockRecommendations = {
      recommendations: [
        {
          id: '1',
          title: 'Test Book',
          author: 'Test Author',
          image_url: '', // Empty image URL
          average_rating: 4.5,
          rating_number: 100,
          price: 9.99,
          description: 'Test description'
        }
      ]
    };

    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockRecommendations)
      })
    );

    render(<Home />);
    
    const form = screen.getByRole('form');
    fireEvent.submit(form);

    // Wait for recommendations to be displayed with fallback cover
    await waitFor(() => {
      // Find all elements containing the book title
      const titleElements = screen.getAllByText('Test Book');
      expect(titleElements.length).toBeGreaterThan(0);

      // Find the dynamic book cover container
      const dynamicCover = screen.getAllByText('Test Book').find(element => {
        const parent = element.parentElement?.parentElement;
        return parent?.style.backgroundColor !== undefined;
      });
      expect(dynamicCover).toBeDefined();
    });
  });
}); 