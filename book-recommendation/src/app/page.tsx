// src/app/page.tsx
"use client";

import { useState, useEffect } from 'react';
import RecommendationCard from '../../components/RecommendationCard';

export default function Home() {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/recs/retrieve?count=10&debug=false', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        accept: 'application/json',
      },
      body: JSON.stringify({
        user_ids_raw: ['AE224PFXAEAT66IXX43GRJSWHXCA'],
        item_seq_raw: [['0439064864', '043935806X']],
        candidate_items_raw: [],
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        setRecommendations(data.recommendations);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching recommendations:', error);
        setLoading(false);
      });
  }, []);

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Book Recommendations</h1>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div style={gridStyle}>
          {recommendations.map((rec, index) => (
            <RecommendationCard key={index} recommendation={rec} />
          ))}
        </div>
      )}
    </div>
  );
}

// Simple grid layout style
const gridStyle = {
  display: 'flex',
  flexWrap: 'wrap' as 'wrap',
  gap: '1rem',
};