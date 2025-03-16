import type { Recommendation } from '@/types/api';
import { MAX_STORAGE_ITEMS } from '@/lib/config/recentlyViewed';

const STORAGE_KEY = 'recently-viewed-books';

// Custom event name for recently viewed changes
export const RECENTLY_VIEWED_CHANGE_EVENT = 'recently-viewed-change';

// Function to dispatch the custom event
function dispatchRecentlyViewedChangeEvent() {
  if (typeof window !== 'undefined') {
    const event = new CustomEvent(RECENTLY_VIEWED_CHANGE_EVENT);
    window.dispatchEvent(event);
  }
}

export interface RecentlyViewedBook extends Recommendation {
  viewedAt: number; // Timestamp when the book was viewed
}

/**
 * Get all recently viewed books from storage
 */
export function getRecentlyViewedBooks(): RecentlyViewedBook[] {
  if (typeof window === 'undefined') {
    return []; // Return empty array on server-side
  }
  
  try {
    const storedBooks = localStorage.getItem(STORAGE_KEY);
    return storedBooks ? JSON.parse(storedBooks) : [];
  } catch (error) {
    console.error('Failed to parse recently viewed books from localStorage', error);
    return [];
  }
}

/**
 * Add a book to recently viewed books
 */
export function addRecentlyViewedBook(book: Recommendation): void {
  if (typeof window === 'undefined') {
    return; // Do nothing on server-side
  }
  
  try {
    const recentlyViewed = getRecentlyViewedBooks();
    
    // Check if the book is already in the list
    const existingIndex = recentlyViewed.findIndex(item => item.parent_asin === book.parent_asin);
    
    // Sanitize book data to ensure all fields have valid values
    const sanitizedBook: Recommendation = {
      parent_asin: book.parent_asin || '',
      title: book.title || '',
      subtitle: book.subtitle || '',
      main_category: book.main_category || 'Unknown',
      average_rating: typeof book.average_rating === 'number' ? book.average_rating : 0,
      rating_number: typeof book.rating_number === 'number' ? book.rating_number : 0,
      price: book.price || null,
      image_url: book.image_url || '',
      score: typeof book.score === 'number' ? book.score : 0
    };
    
    // Create new item with current timestamp
    const newItem: RecentlyViewedBook = {
      ...sanitizedBook,
      viewedAt: Date.now()
    };
    
    let updatedList: RecentlyViewedBook[];
    
    if (existingIndex >= 0) {
      // Remove the existing item
      updatedList = recentlyViewed.filter((_, index) => index !== existingIndex);
    } else {
      updatedList = [...recentlyViewed];
    }
    
    // Add the new item at the beginning
    updatedList.unshift(newItem);
    
    // Limit the list to MAX_ITEMS
    if (updatedList.length > MAX_STORAGE_ITEMS) {
      updatedList = updatedList.slice(0, MAX_STORAGE_ITEMS);
    }
    
    // Save to localStorage
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedList));
    
    // Dispatch the change event
    dispatchRecentlyViewedChangeEvent();
  } catch (error) {
    console.error('Failed to add book to recently viewed', error);
  }
}

/**
 * Clear all recently viewed books
 */
export function clearRecentlyViewedBooks(): void {
  if (typeof window === 'undefined') {
    return; // Do nothing on server-side
  }
  
  localStorage.removeItem(STORAGE_KEY);
  
  // Dispatch the change event
  dispatchRecentlyViewedChangeEvent();
} 