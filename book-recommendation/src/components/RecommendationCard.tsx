// components/RecommendationCard.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { BookCover, isPlaceholderImage } from './BookCover';

export interface Recommendation {
  image_url: string;
  title: string;
  subtitle: string;
  average_rating: number;
  rating_number: number;
  price: string;
  // Add any additional fields you might want to display on the detail page
  description?: string;
  author?: string;
  isbn?: string;
}

const RecommendationCard = ({ recommendation }: { recommendation: Recommendation }) => {
  const router = useRouter();

  // Card styling - now entire card is clickable
  const cardStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    border: '1px solid #ccc',
    borderRadius: '8px',
    padding: '1rem',
    width: '200px',
    minHeight: '340px',
    textAlign: 'center',
    boxShadow: '2px 2px 12px rgba(0,0,0,0.1)',
    cursor: 'pointer',
    transition: 'transform 0.2s, box-shadow 0.2s',
  };

  const hoverStyle: React.CSSProperties = {
    transform: 'translateY(-5px)',
    boxShadow: '2px 5px 15px rgba(0,0,0,0.15)',
  };

  const handleCardClick = () => {
    // Remove any control characters from the recommendation object
    const sanitizedRecommendation = JSON.parse(
      JSON.stringify(recommendation).replace(/[\u0000-\u001F\u007F-\u009F]/g, '')
    );
    
    // Encode the sanitized recommendation data to pass through the URL
    const encodedData = encodeURIComponent(JSON.stringify(sanitizedRecommendation));
    router.push(`/book-details?data=${encodedData}`);
  };

  return (
    <div 
      style={cardStyle} 
      onClick={handleCardClick}
      onMouseOver={(e: React.MouseEvent<HTMLDivElement>) => {
        Object.assign(e.currentTarget.style, hoverStyle);
      }}
      onMouseOut={(e: React.MouseEvent<HTMLDivElement>) => {
        e.currentTarget.style.transform = '';
        e.currentTarget.style.boxShadow = '2px 2px 12px rgba(0,0,0,0.1)';
      }}
    >
      <div>
        {isPlaceholderImage(recommendation.image_url) ? (
          <div style={{ margin: '0 auto', width: '150px', height: '200px' }}>
            <BookCover
              title={recommendation.title}
              author={recommendation.author}
              width={150}
              height={200}
            />
          </div>
        ) : (
          <div style={{ position: 'relative', width: '150px', height: '200px', margin: '0 auto' }}>
            <Image
              src={recommendation.image_url}
              alt={recommendation.title}
              style={{ objectFit: 'cover' }}
              fill={true}
            />
          </div>
        )}
        <h3>{recommendation.title}</h3>
        <p>{recommendation.subtitle}</p>
        <p>
          Rating: {recommendation.average_rating} ({recommendation.rating_number} reviews)
        </p>
        <p>Price: {recommendation.price}</p>
      </div>
    </div>
  );
};

export default RecommendationCard;