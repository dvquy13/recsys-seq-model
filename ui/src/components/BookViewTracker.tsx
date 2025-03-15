'use client';

import { useEffect } from 'react';
import type { Recommendation } from '@/types/api';
import { addRecentlyViewedBook } from '@/lib/recentlyViewed';

interface BookViewTrackerProps {
  book: Recommendation;
}

export function BookViewTracker({ book }: BookViewTrackerProps) {
  useEffect(() => {
    // Make sure the book has all required properties before saving
    if (book && typeof book.title === 'string' && typeof book.parent_asin === 'string') {
      // Ensure all required properties exist with defaults if needed
      const validBook: Recommendation = {
        ...book,
        average_rating: typeof book.average_rating === 'number' ? book.average_rating : 0,
        rating_number: typeof book.rating_number === 'number' ? book.rating_number : 0,
        price: book.price || null,
        subtitle: book.subtitle || '',
        main_category: book.main_category || 'Unknown',
        score: typeof book.score === 'number' ? book.score : 0,
        image_url: book.image_url || ''
      };
      
      // Add the validated book to recently viewed
      addRecentlyViewedBook(validBook);
    }
  }, [book]);

  // This component doesn't render anything
  return null;
} 