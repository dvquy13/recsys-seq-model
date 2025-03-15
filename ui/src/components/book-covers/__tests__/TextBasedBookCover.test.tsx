import React from 'react';
import { render, screen } from '@testing-library/react';
import { TextBasedBookCover } from '../TextBasedBookCover';

describe('TextBasedBookCover', () => {
  const defaultProps = {
    title: 'Test Book',
    width: 200,
    height: 300,
  };

  it('renders with title when no image is provided', () => {
    render(<TextBasedBookCover {...defaultProps} />);
    expect(screen.getByText('Test Book')).toBeInTheDocument();
  });

  it('renders with author when provided and no image', () => {
    render(<TextBasedBookCover {...defaultProps} author="John Doe" />);
    expect(screen.getByText('Test Book')).toBeInTheDocument();
    expect(screen.getByText('by John Doe')).toBeInTheDocument();
  });

  it('truncates long title', () => {
    const longTitle = 'This is a very long book title that should be truncated at some point';
    render(<TextBasedBookCover {...defaultProps} title={longTitle} />);
    expect(screen.getByText(`${longTitle.slice(0, 47)}...`)).toBeInTheDocument();
  });

  it('truncates long author name', () => {
    const longAuthor = 'Dr. John Smith Johnson III PhD from University of Testing';
    render(<TextBasedBookCover {...defaultProps} author={longAuthor} />);
    expect(screen.getByText(`by ${longAuthor.slice(0, 27)}...`)).toBeInTheDocument();
  });

  it('uses custom background color when provided', () => {
    const backgroundColor = '#FF0000';
    render(<TextBasedBookCover {...defaultProps} backgroundColor={backgroundColor} />);
    const card = screen.getByTestId('book-cover');
    expect(card).toHaveStyle({ backgroundColor });
  });

  it('uses consistent background color based on parent_asin', () => {
    const parent_asin = 'B000Q9INGI';
    const { container: firstRender } = render(
      <TextBasedBookCover {...defaultProps} parent_asin={parent_asin} />
    );
    const { container: secondRender } = render(
      <TextBasedBookCover {...defaultProps} parent_asin={parent_asin} />
    );
    
    const firstColor = firstRender.querySelector('[data-testid="book-cover"]')?.getAttribute('style');
    const secondColor = secondRender.querySelector('[data-testid="book-cover"]')?.getAttribute('style');
    expect(firstColor).toBe(secondColor);
  });

  it('renders image when valid image URL is provided', () => {
    const imageUrl = 'https://example.com/book.jpg';
    render(<TextBasedBookCover {...defaultProps} imageUrl={imageUrl} />);
    const image = screen.getByAltText('Test Book');
    expect(image).toBeInTheDocument();
    expect(image).toHaveAttribute('src');
  });

  it('does not render title when valid image is provided', () => {
    const imageUrl = 'https://example.com/book.jpg';
    render(<TextBasedBookCover {...defaultProps} imageUrl={imageUrl} />);
    expect(screen.queryByText('Test Book')).not.toBeInTheDocument();
  });

  it('renders title when placeholder image URL is provided', () => {
    const imageUrl = 'https://placehold.co/400x600';
    render(<TextBasedBookCover {...defaultProps} imageUrl={imageUrl} />);
    expect(screen.getByText('Test Book')).toBeInTheDocument();
  });
}); 