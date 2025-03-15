import React, { useEffect, useState } from 'react';
import { RecommendationsGrid } from './RecommendationsGrid';
import type { Recommendation } from '@/types/api';
import { getRecentlyViewedBooks, type RecentlyViewedBook } from '@/lib/recentlyViewed';

export function RecentlyViewedGrid() {
  const [recentlyViewed, setRecentlyViewed] = useState<Recommendation[]>([]);
  
  useEffect(() => {
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
    
    setRecentlyViewed(recommendations);
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