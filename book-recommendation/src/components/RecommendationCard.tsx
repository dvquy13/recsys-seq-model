// components/RecommendationCard.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { BookCover, isPlaceholderImage } from './BookCover';
import type { BookRecommendation } from '@/types/api';

interface RecommendationCardProps {
  recommendation: BookRecommendation;
}

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation }) => {
  const router = useRouter();

  const {
    title,
    price,
    average_rating,
    rating_number,
    image_url,
    subtitle,
    main_category,
    score,
    parent_asin
  } = recommendation;

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
      const imageUrl = isPlaceholderImage(image_url) 
        ? '' // Empty URL will trigger BookCover in the details view
        : image_url;

      // Pass through all fields from the recommendation
      const validatedRecommendation = {
        image_url: imageUrl,
        title: title || 'Untitled',
        subtitle: subtitle || '',
        average_rating: Number(average_rating) || 0,
        rating_number: Number(rating_number) || 0,
        price: String(price || '0'),
        main_category: main_category || '',
        score: Number(score) || 0,
        parent_asin: parent_asin || ''
      };
      
      // Encode the data for URL
      const encodedData = encodeURIComponent(JSON.stringify(validatedRecommendation));
      
      router.push(`/book-details?data=${encodedData}`);
    } catch (error) {
      console.error('Error preparing book data:', error);
      // Provide a fallback with minimal data
      const fallbackData = {
        image_url: '',
        title: title || 'Untitled',
        subtitle: subtitle || '',
        average_rating: 0,
        rating_number: 0,
        price: '0',
        main_category: '',
        score: 0,
        parent_asin: ''
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
        {image_url.includes('placehold.co') ? (
          <BookCover title={title} width={200} height={300} />
        ) : (
          <div style={{ 
            position: 'relative', 
            width: '200px', 
            height: '300px', 
            margin: '0 auto',
            marginBottom: '20px',
            overflow: 'hidden',
            borderRadius: '4px'
          }}>
            <Image
              src={image_url || '/placeholder.jpg'}
              alt={title}
              style={{ 
                objectFit: 'cover',
                width: '100%',
                height: '100%'
              }}
              fill={true}
              sizes="200px"
            />
          </div>
        )}
        <div style={{ marginTop: '10px' }}>
          <h3 style={{ marginTop: '0', fontSize: '1rem', fontWeight: 'bold' }}>{title}</h3>
          {average_rating && (
            <p style={{ margin: '0.25rem 0', fontSize: '0.9rem' }}>
              Rating: {average_rating.toFixed(1)} ({rating_number} reviews)
            </p>
          )}
          {price && price !== "None" && (
            <p style={{ margin: '0.25rem 0', fontSize: '0.9rem' }}>
              Price: ${price}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecommendationCard;