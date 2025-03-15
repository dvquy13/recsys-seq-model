// src/app/page.tsx
"use client";

import { useState } from 'react';
import Header from '../components/Header';
import RecommendationCard from '../components/RecommendationCard';

export default function Home() {
  const [userId, setUserId] = useState('AE224PFXAEAT66IXX43GRJSWHXCA');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchRecommendations = (userId: string) => {
    setLoading(true);
    fetch('http://localhost:8000/recs/retrieve?count=10&debug=false', {
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
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (userId.trim() !== '') {
      fetchRecommendations(userId);
    }
  };

  return (
    <div style={{ padding: '2rem' }}>
      <Header title="Book Recommendations" />
      <form onSubmit={handleSubmit} style={{ marginBottom: '1rem' }}>
        <input
          type="text"
          placeholder="Enter User ID"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          style={{ padding: '0.5rem', fontSize: '1rem', width: '300px' }}
        />
        <button
          type="submit"
          style={{ padding: '0.5rem 1rem', marginLeft: '0.5rem', fontSize: '1rem' }}
        >
          Submit
        </button>
      </form>
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