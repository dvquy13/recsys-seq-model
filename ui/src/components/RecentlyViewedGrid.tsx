import React from 'react';
import { RecommendationsGrid } from './RecommendationsGrid';
import { useAppState } from '@/providers/app-state-provider';

export function RecentlyViewedGrid() {
  // Get the recently viewed books from the context
  const { recentlyViewedBooks } = useAppState();
  
  // Don't render if there are no recently viewed books
  if (recentlyViewedBooks.length === 0) {
    return null;
  }
  
  return (
    <RecommendationsGrid
      title="Recently Viewed"
      recommendations={recentlyViewedBooks}
      emptyMessage="No recently viewed books"
      className="mt-6 mb-8 space-y-4" // Add margin bottom to separate from the recommendations grid
    />
  );
} 