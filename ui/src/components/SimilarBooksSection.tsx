'use client';

import { useState, useEffect } from 'react';
import { recommendationsApi } from '@/lib/api';
import { RecommendationsGrid } from '@/components/RecommendationsGrid';
import type { Recommendation } from '@/types/api';
import { Loader2 } from 'lucide-react';

interface SimilarBooksSectionProps {
  bookId: string;
}

export function SimilarBooksSection({ bookId }: SimilarBooksSectionProps) {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchRecommendations() {
      if (!bookId) {
        setLoading(false);
        return;
      }
      
      try {
        // Call the API with item_seq_raw containing just the current book
        const response = await recommendationsApi.getRecommendations(
          "anonymous", // userId (could be enhanced to use actual user)
          10, // count
          false, // debug
          [[bookId]] // item_seq_raw with the current book id
        );
        
        if (response.recommendations && response.recommendations.length > 0) {
          setRecommendations(response.recommendations);
        }
        setLoading(false);
      } catch (err) {
        console.error('Error fetching recommendations:', err);
        setError('Failed to load recommendations');
        setLoading(false);
      }
    }

    fetchRecommendations();
  }, [bookId]);

  if (loading) {
    return (
      <div className="mt-6 flex justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-primary" data-testid="loading-spinner" />
      </div>
    );
  }

  if (error) {
    return <div className="mt-6 text-center text-red-500">{error}</div>;
  }

  if (recommendations.length === 0) {
    return null;
  }

  return (
    <RecommendationsGrid
      title="Similar Books You Might Like"
      recommendations={recommendations}
    />
  );
}