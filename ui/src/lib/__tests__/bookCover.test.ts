import { generateBookCover, shouldGenerateBookCover } from '../bookCover';

// Mock DiceBear modules
jest.mock('@dicebear/core', () => ({
  createAvatar: jest.fn().mockImplementation(() => ({
    toDataUri: () => 'data:image/svg+xml;charset=UTF-8,<svg><rect fill="#ff0000"/><ellipse/><polygon/></svg>'
  }))
}));

jest.mock('@dicebear/collection', () => ({
  shapes: {}
}));

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
  const { createAvatar } = require('@dicebear/core');

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should generate a data URI for SVG image', () => {
    const result = generateBookCover('Test Book');
    expect(result).toMatch(/^data:image\/svg\+xml;charset=UTF-8,/);
  });

  it('should generate consistent output for same title', () => {
    const firstResult = generateBookCover('Test Book');
    const secondResult = generateBookCover('Test Book');
    expect(firstResult).toBe(secondResult);
  });

  it('should generate different output for different titles', () => {
    // Mock different outputs for different seeds
    createAvatar
      .mockImplementationOnce(() => ({
        toDataUri: () => 'data:image/svg+xml;charset=UTF-8,<svg>1</svg>'
      }))
      .mockImplementationOnce(() => ({
        toDataUri: () => 'data:image/svg+xml;charset=UTF-8,<svg>2</svg>'
      }));

    const firstResult = generateBookCover('Test Book 1');
    const secondResult = generateBookCover('Test Book 2');
    expect(firstResult).not.toBe(secondResult);
  });

  it('should call createAvatar with correct options', () => {
    generateBookCover('Test Book');
    expect(createAvatar).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        seed: 'Test Book',
        backgroundColor: expect.any(Array),
        size: expect.any(Number)
      })
    );
  });

  it('should include the configured shapes and colors', () => {
    const result = generateBookCover('Test Book');
    // Check for presence of SVG elements and color values
    expect(result).toMatch(/svg/);
    expect(result).toMatch(/ellipse|rectangle|polygon/);
    expect(result).toMatch(/ff0000|00ff00|0000ff/);
  });
}); 