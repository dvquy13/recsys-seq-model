import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { AppStateProvider, useAppState } from '../app-state-provider';
import '@testing-library/jest-dom';

// Mock localStorage
const mockLocalStorage = (() => {
  let store: Record<string, string> = {};
  
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    }
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage
});

// Mock the API
jest.mock('@/lib/api', () => ({
  recommendationsApi: {
    getRecommendations: jest.fn().mockResolvedValue({
      recommendations: [
        {
          parent_asin: 'test-asin-1',
          title: 'Test Book 1',
          average_rating: 4.5,
          rating_number: 100,
          price: 9.99,
          subtitle: 'Test Subtitle',
          main_category: 'Fiction',
          score: 0.95,
          image_url: 'https://example.com/image1.jpg'
        }
      ]
    })
  }
}));

// Mock custom events
global.CustomEvent = class MockCustomEvent<T = any> extends Event {
  detail?: T;
  
  constructor(name: string, options?: CustomEventInit<T>) {
    super(name, options);
    this.detail = options?.detail;
  }
  
  initCustomEvent(type: string, bubbles?: boolean, cancelable?: boolean, detail?: T): void {
    this.detail = detail;
  }
} as any;

// Test component that uses the context
const TestComponent = () => {
  const { 
    userId, 
    setUserId, 
    submitUserId, 
    recentlyViewedBooks,
    addBookToRecentlyViewed
  } = useAppState();
  
  return (
    <div>
      <div data-testid="user-id">{userId}</div>
      <div data-testid="recently-viewed-count">{`${recentlyViewedBooks.length}`}</div>
      <button 
        data-testid="set-user-id" 
        onClick={() => setUserId('test-user')}
      >
        Set User ID
      </button>
      <button 
        data-testid="submit-user-id" 
        onClick={() => submitUserId('test-user')}
      >
        Submit User ID
      </button>
      <button 
        data-testid="add-book" 
        onClick={() => addBookToRecentlyViewed({
          parent_asin: 'test-book',
          title: 'Test Book',
          average_rating: 4.0,
          rating_number: 100,
          price: '9.99',
          subtitle: 'Test Subtitle',
          main_category: 'Fiction',
          score: 0.9,
          image_url: 'https://example.com/test.jpg'
        })}
      >
        Add Book
      </button>
    </div>
  );
};

describe('AppStateProvider', () => {
  beforeEach(() => {
    mockLocalStorage.clear();
    jest.clearAllMocks();
  });
  
  it('provides user ID state and functions', async () => {
    render(
      <AppStateProvider>
        <TestComponent />
      </AppStateProvider>
    );
    
    // Initial state
    expect(screen.getByTestId('user-id')).toHaveTextContent('');
    
    // Set user ID
    fireEvent.click(screen.getByTestId('set-user-id'));
    expect(screen.getByTestId('user-id')).toHaveTextContent('test-user');
    
    // Submit user ID
    await act(async () => {
      fireEvent.click(screen.getByTestId('submit-user-id'));
    });
    
    // Check localStorage
    expect(mockLocalStorage.getItem('last-submitted-user-id')).toBe('test-user');
  });
  
  it('manages recently viewed books', async () => {
    render(
      <AppStateProvider>
        <TestComponent />
      </AppStateProvider>
    );
    
    // Initial state
    expect(screen.getByTestId('recently-viewed-count')).toHaveTextContent('0');
    
    // Add a book
    await act(async () => {
      fireEvent.click(screen.getByTestId('add-book'));
    });
    
    // Check if book was added
    await waitFor(() => {
      expect(screen.getByTestId('recently-viewed-count')).toHaveTextContent('1');
    });
  });
}); 