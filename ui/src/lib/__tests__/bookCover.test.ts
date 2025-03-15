import { generateBookCover, shouldGenerateBookCover } from '../bookCover';

describe('shouldGenerateBookCover', () => {
  it('should return true when image_url is undefined', () => {
    expect(shouldGenerateBookCover(undefined)).toBe(true);
  });

  it('should return true when image_url contains placehold.co', () => {
    expect(shouldGenerateBookCover('https://placehold.co/400')).toBe(true);
    expect(shouldGenerateBookCover('http://placehold.co/600x400')).toBe(true);
  });

  it('should return false for valid image URLs', () => {
    expect(shouldGenerateBookCover('https://example.com/book.jpg')).toBe(false);
    expect(shouldGenerateBookCover('https://images.amazon.com/123.png')).toBe(false);
  });
});

describe('generateBookCover', () => {
  it('should generate a placeholder URL with the book title', () => {
    const result = generateBookCover('Test Book');
    expect(result).toMatch(/^https:\/\/placehold\.co\/400x600/);
    expect(result).toContain('text=Test%20Book');
  });

  it('should generate consistent output for same title', () => {
    const firstResult = generateBookCover('Test Book');
    const secondResult = generateBookCover('Test Book');
    expect(firstResult).toBe(secondResult);
  });

  it('should generate different output for different titles', () => {
    const firstResult = generateBookCover('Test Book 1');
    const secondResult = generateBookCover('Test Book 2');
    expect(firstResult).not.toBe(secondResult);
  });

  it('should properly encode special characters in title', () => {
    const result = generateBookCover('Test & Book');
    expect(result).toContain('text=Test%20%26%20Book');
  });
}); 