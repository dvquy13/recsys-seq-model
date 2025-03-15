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
    try {
      // For placeholder images, generate a BookCover-style URL
      const imageUrl = isPlaceholderImage(recommendation.image_url) 
        ? '' // Empty URL will trigger BookCover in the details view
        : recommendation.image_url;

      // Ensure all required fields are present and have the correct type
      const validatedRecommendation = {
        image_url: imageUrl,
        title: recommendation.title || 'Untitled',
        subtitle: recommendation.subtitle || '',
        average_rating: Number(recommendation.average_rating) || 0,
        rating_number: Number(recommendation.rating_number) || 0,
        price: String(recommendation.price || '0'),
        description: recommendation.description,
        author: recommendation.author,
        isbn: recommendation.isbn,
      };
      
      // Encode the data for URL
      const encodedData = encodeURIComponent(JSON.stringify(validatedRecommendation));
      
      router.push(`/book-details?data=${encodedData}`);
    } catch (error) {
      console.error('Error preparing book data:', error);
      // Provide a fallback with minimal data
      const fallbackData = {
        image_url: '',
        title: recommendation.title || 'Untitled',
        subtitle: '',
        average_rating: 0,
        rating_number: 0,
        price: '0',
      };
      const encodedFallback = encodeURIComponent(JSON.stringify(fallbackData));
      router.push(`/book-details?data=${encodedFallback}`);
    }
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
          <div style={{ 
            margin: '0 auto', 
            width: '150px', 
            height: '200px', 
            marginBottom: '20px' 
          }}>
            <BookCover
              title={recommendation.title}
              author={recommendation.author}
              width={150}
              height={200}
            />
          </div>
        ) : (
          <div style={{ 
            position: 'relative', 
            width: '150px', 
            height: '200px', 
            margin: '0 auto',
            marginBottom: '20px',
            overflow: 'hidden',
            borderRadius: '4px'
          }}>
            <Image
              src={recommendation.image_url}
              alt={recommendation.title}
              style={{ 
                objectFit: 'cover',
                width: '100%',
                height: '100%'
              }}
              fill={true}
              sizes="150px"
            />
          </div>
        )}
        <div style={{ marginTop: '10px' }}>
          <h3 style={{ marginTop: '0', fontSize: '1rem', fontWeight: 'bold' }}>{recommendation.title}</h3>
          <p>{recommendation.subtitle}</p>
          <p>
            Rating: {recommendation.average_rating} ({recommendation.rating_number} reviews)
          </p>
          <p>Price: {recommendation.price}</p>
        </div>
      </div>
    </div>
  );
};

export default RecommendationCard;