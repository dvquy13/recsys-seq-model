// components/RecommendationCard.js
import React from 'react';

interface Recommendation {
  image_url: string;
  title: string;
  subtitle: string;
  average_rating: number;
  rating_number: number;
  price: string;
}

const RecommendationCard = ({ recommendation }: { recommendation: Recommendation }) => {
  return (
    <div style={cardStyle}>
      <img
        src={recommendation.image_url}
        alt={recommendation.title}
        style={{ width: '150px', height: '200px', objectFit: 'cover' }}
      />
      <h3>{recommendation.title}</h3>
      <p>{recommendation.subtitle}</p>
      <p>
        Rating: {recommendation.average_rating} ({recommendation.rating_number} reviews)
      </p>
      <p>Price: {recommendation.price}</p>
    </div>
  );
};

// Simple inline style for the card
const cardStyle = {
  border: '1px solid #ccc',
  borderRadius: '8px',
  padding: '1rem',
  width: '200px',
  textAlign: 'center' as const, // Use the 'as const' assertion to tell TypeScript that this is a literal value
  boxShadow: '2px 2px 12px rgba(0,0,0,0.1)',
};

export default RecommendationCard;
