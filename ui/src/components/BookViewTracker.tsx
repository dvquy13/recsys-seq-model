'use client';

import { useEffect } from 'react';
import type { Recommendation } from '@/types/api';
import { addRecentlyViewedBook } from '@/lib/recentlyViewed';
import { updatePersonalizedRecs } from '@/lib/personalizedRecs';

interface BookViewTrackerProps {
  book: Recommendation;
}

export function BookViewTracker({ book }: BookViewTrackerProps) {
  useEffect(() => {
    const trackBookView = async () => {
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
        
        // Get the current user ID from localStorage
        const userId = localStorage.getItem('last-submitted-user-id');
        
        // Update personalized recommendations even if userId is null/empty
        // This will ensure item_seq_raw is updated even for guest users
        updatePersonalizedRecs(userId || 'guest').catch(error => {
          console.error('Failed to update personalized recommendations:', error);
        });
      }
    };
    
    // Execute the tracking function
    trackBookView();
  }, [book]);

  // This component doesn't render anything
  return null;
} 