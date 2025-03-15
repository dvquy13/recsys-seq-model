import React, { useEffect, useState } from 'react';
import { RecommendationsGrid } from './RecommendationsGrid';
import type { Recommendation } from '@/types/api';
import { 
  getRecentlyViewedBooks, 
  type RecentlyViewedBook,
  RECENTLY_VIEWED_CHANGE_EVENT 
} from '@/lib/recentlyViewed';

// Maximum number of recently viewed items to display
const MAX_DISPLAY_ITEMS = 3;

// Storage key used in recentlyViewed.ts
const STORAGE_KEY = 'recently-viewed-books';

export function RecentlyViewedGrid() {
  const [recentlyViewed, setRecentlyViewed] = useState<Recommendation[]>([]);

  const loadRecentlyViewed = () => {
    // Get recently viewed books and sort by viewedAt (most recent first)
    const books = getRecentlyViewedBooks()
      .sort((a, b) => b.viewedAt - a.viewedAt);
    
    // Convert RecentlyViewedBook[] to Recommendation[] and ensure all required fields exist
    const recommendations: Recommendation[] = books
      .filter(book => {
        // Filter out any books with missing required properties
        return book && 
          typeof book.title === 'string' && 
          typeof book.parent_asin === 'string' &&
          typeof book.average_rating === 'number' && 
          typeof book.rating_number === 'number';
      })
      .map(({ viewedAt, ...book }) => ({
        ...book,
        // Ensure all required properties are valid
        average_rating: book.average_rating || 0,
        rating_number: book.rating_number || 0,
        price: book.price || null,
        subtitle: book.subtitle || '',
        main_category: book.main_category || 'Unknown',
        score: book.score || 0
      }));
    
    // Only take the first MAX_DISPLAY_ITEMS
    setRecentlyViewed(recommendations.slice(0, MAX_DISPLAY_ITEMS));
  };
  
  useEffect(() => {
    // Load recently viewed books initially
    loadRecentlyViewed();
    
    // Set up listeners for both storage events and our custom event
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY || event.key === null) {
        loadRecentlyViewed();
      }
    };
    
    const handleRecentlyViewedChange = () => {
      loadRecentlyViewed();
    };
    
    // Add event listeners
    window.addEventListener('storage', handleStorageChange);
    window.addEventListener(RECENTLY_VIEWED_CHANGE_EVENT, handleRecentlyViewedChange);
    
    // Clean up event listeners on unmount
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener(RECENTLY_VIEWED_CHANGE_EVENT, handleRecentlyViewedChange);
    };
  }, []);
  
  // Don't render if there are no recently viewed books
  if (recentlyViewed.length === 0) {
    return null;
  }
  
  return (
    <RecommendationsGrid
      title="Recently Viewed"
      recommendations={recentlyViewed}
      emptyMessage="No recently viewed books"
      className="mt-6 mb-8 space-y-4" // Add margin bottom to separate from the recommendations grid
    />
  );
} 