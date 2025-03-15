// src/app/page.tsx
"use client";

import React from 'react';
import { useState } from 'react';
import Header from '@/components/Header';
import RecommendationCard from '@/components/RecommendationCard';
import type { BookRecommendation, RecommendationResponse } from '@/types/api';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function Home() {
  const [userId, setUserId] = useState('AE224PFXAEAT66IXX43GRJSWHXCA');
  const [recommendations, setRecommendations] = useState<BookRecommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRecommendations = async (userId: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/recs/retrieve?count=10&debug=false', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          accept: 'application/json',
        },
        body: JSON.stringify({
          user_ids_raw: [userId],
          item_seq_raw: [['0439064864', '043935806X']],
          candidate_items_raw: [],
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('API Response:', data); // Debug log
      
      // Check if data has the expected structure
      if (!data || !Array.isArray(data.recommendations)) {
        throw new Error('Invalid response format from API');
      }
      
      setRecommendations(data.recommendations);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch recommendations');
      setRecommendations([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (userId.trim() !== '') {
      fetchRecommendations(userId);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header title="Book Recommendations" />
      <main className="container py-6">
        <div className="flex flex-col items-center space-y-8">
          <form onSubmit={handleSubmit} className="flex w-full max-w-sm items-center space-x-2">
            <Input
              type="text"
              placeholder="Enter User ID"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
            <Button type="submit">
              Get Recommendations
            </Button>
          </form>
          
          {error && (
            <div className="text-red-500">
              {error}
            </div>
          )}
          
          {loading ? (
            <div className="flex h-[200px] items-center justify-center">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </div>
          ) : recommendations.length > 0 ? (
            <div className="w-full overflow-hidden">
              <div className="flex snap-x snap-mandatory gap-6 overflow-x-auto pb-6 scrollbar-hide">
                {recommendations.map((recommendation, index) => (
                  <div key={index} className="snap-start shrink-0 first:pl-4 last:pr-4">
                    <RecommendationCard recommendation={recommendation} />
                  </div>
                ))}
              </div>
            </div>
          ) : !error && (
            <div className="text-muted-foreground">
              No recommendations available. Try searching for a different user ID.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}