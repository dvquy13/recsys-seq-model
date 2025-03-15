import React from 'react';
import type { Recommendation } from '@/types/api';
import { RecommendationCard } from './RecommendationCard';

export interface RecommendationsGridProps {
  title: string;
  recommendations: Recommendation[];
  emptyMessage?: string;
  className?: string;
  gridClassName?: string;
  titleClassName?: string;
}

export function RecommendationsGrid({
  title,
  recommendations,
  emptyMessage = 'No recommendations available',
  className = 'mt-6 space-y-4',
  gridClassName = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4',
  titleClassName = 'text-lg font-semibold',
}: RecommendationsGridProps) {
  return (
    <div className={className}>
      <h3 className={titleClassName}>{title}</h3>
      
      {recommendations.length === 0 ? (
        <p className="text-muted-foreground">{emptyMessage}</p>
      ) : (
        <div className={gridClassName}>
          {recommendations.map((recommendation, index) => (
            <RecommendationCard 
              key={`${recommendation.parent_asin}-${index}`} 
              recommendation={recommendation}
              showScore={false}
            />
          ))}
        </div>
      )}
    </div>
  );
} 