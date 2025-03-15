import React from 'react';
import { render, screen } from '@testing-library/react';
import { BookCover, implementations } from '..';

describe('BookCover', () => {
  const defaultProps = {
    title: 'Test Book',
    width: 200,
    height: 300,
  };

  it('uses textBased implementation by default', () => {
    render(<BookCover {...defaultProps} />);
    expect(screen.getByText('Test Book')).toBeInTheDocument();
  });

  it('switches between implementations correctly', () => {
    // Test each implementation
    implementations.forEach(impl => {
      const { container } = render(
        <BookCover {...defaultProps} implementation={impl.name} />
      );
      
      // All implementations should have the book-cover test id
      expect(container.querySelector('[data-testid="book-cover"]')).toBeInTheDocument();
    });
  });

  it('falls back to first implementation for invalid implementation name', () => {
    render(<BookCover {...defaultProps} implementation="invalid" />);
    // Should render textBased implementation (first in array)
    expect(screen.getByText('Test Book')).toBeInTheDocument();
  });

  it('handles image URLs consistently across implementations', () => {
    const imageUrl = 'https://example.com/book.jpg';
    
    // Test each implementation with an image
    implementations.forEach(impl => {
      const { container } = render(
        <BookCover {...defaultProps} implementation={impl.name} imageUrl={imageUrl} />
      );
      
      const image = container.querySelector('img');
      expect(image).toBeInTheDocument();
      expect(image).toHaveAttribute('src');
    });
  });

  it('handles placeholder images consistently across implementations', () => {
    const placeholderUrl = 'https://placehold.co/400x600';
    
    // Test each implementation with a placeholder image
    implementations.forEach(impl => {
      render(
        <BookCover {...defaultProps} implementation={impl.name} imageUrl={placeholderUrl} />
      );
      
      // All implementations should show the title for placeholder images
      expect(screen.getByText('Test Book')).toBeInTheDocument();
    });
  });
}); 