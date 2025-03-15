// src/app/page.tsx
"use client";

import React from 'react';
import { useState } from 'react';
import Header from '@/components/Header';
import RecommendationCard from '@/components/RecommendationCard';
import type { BookRecommendation, RecommendationResponse } from '@/types/api';

export default function Home() {
  const [userId, setUserId] = useState('AE224PFXAEAT66IXX43GRJSWHXCA');
  const [recommendations, setRecommendations] = useState<BookRecommendation[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchRecommendations = async (userId: string) => {
    setLoading(true);
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
      
      const data: RecommendationResponse = await response.json();
      setRecommendations(data.recommendations);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
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
    <main>
      <Header title="Book Recommendations" />
      <div style={{ padding: '2rem' }}>
        <form onSubmit={handleSubmit} role="form" style={{ marginBottom: '1rem' }}>
          <input
            type="text"
            placeholder="Enter User ID"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            style={{ 
              padding: '0.5rem', 
              fontSize: '1rem', 
              width: '300px'
            }}
          />
          <button
            type="submit"
            style={{ 
              padding: '0.5rem 1rem', 
              marginLeft: '0.5rem', 
              fontSize: '1rem'
            }}
          >
            Get Recommendations
          </button>
        </form>
        {loading ? (
          <p>Loading recommendations...</p>
        ) : (
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '2rem',
            justifyContent: 'center',
          }}>
            {recommendations.map((recommendation, index) => (
              <RecommendationCard key={index} recommendation={recommendation} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}