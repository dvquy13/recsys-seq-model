import {
  getRecentlyViewedBooks,
  addRecentlyViewedBook,
  clearRecentlyViewedBooks,
  RECENTLY_VIEWED_CHANGE_EVENT
} from '../recentlyViewed';
import { Recommendation } from '@/types/api';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    })
  };
})();

// Mock CustomEvent
class CustomEventMock {
  constructor(public type: string) {}
}

// Mock window.dispatchEvent
const dispatchEventMock = jest.fn();

describe('Recently Viewed Books Utility', () => {
  beforeAll(() => {
    // Mock localStorage
    Object.defineProperty(window, 'localStorage', { value: localStorageMock });
    
    // Mock CustomEvent
    global.CustomEvent = CustomEventMock as any;
    
    // Mock window.dispatchEvent
    Object.defineProperty(window, 'dispatchEvent', { value: dispatchEventMock });
  });

  beforeEach(() => {
    // Clear mocks and localStorage before each test
    jest.clearAllMocks();
    localStorageMock.clear();
  });

  describe('getRecentlyViewedBooks', () => {
    it('should return an empty array when localStorage is empty', () => {
      const result = getRecentlyViewedBooks();
      expect(result).toEqual([]);
      expect(localStorageMock.getItem).toHaveBeenCalledWith('recently-viewed-books');
    });

    it('should parse and return books from localStorage', () => {
      const mockBooks = [
        { parent_asin: '123', title: 'Book 1', viewedAt: 1000 },
        { parent_asin: '456', title: 'Book 2', viewedAt: 2000 }
      ];
      localStorageMock.setItem('recently-viewed-books', JSON.stringify(mockBooks));
      
      const result = getRecentlyViewedBooks();
      expect(result).toEqual(mockBooks);
    });

    it('should return an empty array if localStorage throws an error', () => {
      // Make getItem throw an error
      localStorageMock.getItem.mockImplementationOnce(() => {
        throw new Error('Storage error');
      });
      
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      const result = getRecentlyViewedBooks();
      expect(result).toEqual([]);
      expect(consoleSpy).toHaveBeenCalled();
      
      consoleSpy.mockRestore();
    });
  });

  describe('addRecentlyViewedBook', () => {
    const mockBook: Recommendation = {
      parent_asin: '123',
      title: 'Test Book',
      subtitle: 'A Test Subtitle',
      main_category: 'Fiction',
      average_rating: 4.5,
      rating_number: 100,
      price: '9.99',
      image_url: 'https://example.com/image.jpg',
      score: 0.95
    };

    it('should add a book to localStorage', () => {
      addRecentlyViewedBook(mockBook);

      expect(localStorageMock.setItem).toHaveBeenCalled();
      
      // Verify the book was added with viewedAt timestamp
      const storedValue = JSON.parse(localStorageMock.setItem.mock.calls[0][1]);
      expect(storedValue).toHaveLength(1);
      expect(storedValue[0].parent_asin).toBe(mockBook.parent_asin);
      expect(storedValue[0].title).toBe(mockBook.title);
      expect(storedValue[0].viewedAt).toBeDefined();
    });

    it('should dispatch a change event when a book is added', () => {
      addRecentlyViewedBook(mockBook);
      
      expect(dispatchEventMock).toHaveBeenCalled();
      const event = dispatchEventMock.mock.calls[0][0];
      expect(event.type).toBe(RECENTLY_VIEWED_CHANGE_EVENT);
    });

    it('should move an existing book to the top of the list', () => {
      // Add two books
      addRecentlyViewedBook({ ...mockBook, parent_asin: '123' });
      addRecentlyViewedBook({ ...mockBook, parent_asin: '456' });
      
      // Re-add the first book
      addRecentlyViewedBook({ ...mockBook, parent_asin: '123' });
      
      // Check the order in localStorage
      const storedValue = JSON.parse(localStorageMock.setItem.mock.calls[2][1]);
      expect(storedValue[0].parent_asin).toBe('123'); // Should be first now
      expect(storedValue[1].parent_asin).toBe('456');
    });

    it('should limit the list to MAX_ITEMS', () => {
      // Add 11 different books (MAX_ITEMS is 10)
      for (let i = 1; i <= 11; i++) {
        addRecentlyViewedBook({ ...mockBook, parent_asin: `${i}` });
      }
      
      // Check that only 10 books are stored
      const storedValue = JSON.parse(localStorageMock.setItem.mock.calls[10][1]);
      expect(storedValue).toHaveLength(10);
      expect(storedValue[0].parent_asin).toBe('11'); // Most recent first
      expect(storedValue[9].parent_asin).toBe('2'); // Oldest kept
      // Book with parent_asin '1' should be dropped
      expect(storedValue.find((book: any) => book.parent_asin === '1')).toBeUndefined();
    });

    it('should sanitize book data before saving', () => {
      // Book with missing or undefined values
      const incompleteBook = {
        parent_asin: '123',
        title: 'Test Book',
        // Missing other fields
      } as unknown as Recommendation;
      
      addRecentlyViewedBook(incompleteBook);
      
      const storedValue = JSON.parse(localStorageMock.setItem.mock.calls[0][1])[0];
      
      // Check that all required fields are present with default values
      expect(storedValue.parent_asin).toBe('123');
      expect(storedValue.title).toBe('Test Book');
      expect(storedValue.subtitle).toBe('');
      expect(storedValue.main_category).toBe('Unknown');
      expect(storedValue.average_rating).toBe(0);
      expect(storedValue.rating_number).toBe(0);
      expect(storedValue.price).toBe(null);
      expect(storedValue.image_url).toBe('');
      expect(storedValue.score).toBe(0);
    });
  });

  describe('clearRecentlyViewedBooks', () => {
    it('should remove the item from localStorage', () => {
      clearRecentlyViewedBooks();
      
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('recently-viewed-books');
    });

    it('should dispatch a change event when books are cleared', () => {
      clearRecentlyViewedBooks();
      
      expect(dispatchEventMock).toHaveBeenCalled();
      const event = dispatchEventMock.mock.calls[0][0];
      expect(event.type).toBe(RECENTLY_VIEWED_CHANGE_EVENT);
    });
  });
}); 