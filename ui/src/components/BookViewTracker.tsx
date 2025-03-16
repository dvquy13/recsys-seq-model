'use client';

import { useEffect, useRef } from 'react';
import type { Recommendation } from '@/types/api';
import { useAppState } from '@/providers/app-state-provider';

interface BookViewTrackerProps {
  book: Recommendation;
}

export function BookViewTracker({ book }: BookViewTrackerProps) {
  const { addBookToRecentlyViewed } = useAppState();
  const hasTrackedRef = useRef(false);

  useEffect(() => {
    // Only track the book view once
    if (hasTrackedRef.current) return;
    
    const trackBookView = async () => {
      // Make sure the book has all required properties before saving
      if (book && typeof book.title === 'string' && typeof book.parent_asin === 'string') {
        // Use the context method to add the book to recently viewed
        addBookToRecentlyViewed(book);
        hasTrackedRef.current = true;
      }
    };
    
    // Execute the tracking function
    trackBookView();
  }, [book.parent_asin]); // Only depend on the book ID, not the entire book object or functions

  // This component doesn't render anything
  return null;
} 