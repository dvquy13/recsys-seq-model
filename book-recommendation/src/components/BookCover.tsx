'use client';

import React from 'react';
import Image from 'next/image';

interface BookCoverProps {
  title: string;
  author?: string;
  width?: number;
  height?: number;
  backgroundColor?: string;
  textColor?: string;
  imageUrl: string;
  parent_asin?: string;
}

const colors = [
  '#2D3047', // Dark blue
  '#419D78', // Green
  '#E0A458', // Orange
  '#8D5B4C', // Brown
  '#562C2C', // Dark red
  '#2C365E', // Navy
  '#484D6D', // Slate
] as const;

const getColorFromAsin = (parent_asin: string | undefined, title: string) => {
  // Use parent_asin if available, otherwise fallback to title
  const stringToHash = parent_asin || title;
  
  // Create a simple hash from the string
  const hash = stringToHash.split('').reduce((acc, char) => {
    return ((acc << 5) - acc) + char.charCodeAt(0);
  }, 0);
  
  // Use the absolute value of hash to ensure positive number
  const positiveHash = Math.abs(hash);
  // Get a consistent index within the colors array length
  const colorIndex = positiveHash % colors.length;
  return colors[colorIndex];
};

const truncateText = (text: string, maxLength: number) => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
};

export const BookCover: React.FC<BookCoverProps> = ({
  title,
  author,
  width = 200,
  height = 300,
  backgroundColor,
  textColor = '#FFFFFF',
  imageUrl,
  parent_asin,
}: BookCoverProps) => {
  const bgColor = backgroundColor || getColorFromAsin(parent_asin, title);
  const truncatedTitle = truncateText(title, 50);
  const truncatedAuthor = author ? truncateText(author, 30) : '';
  const patternId = `pattern-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div
      style={{
        width,
        height,
        backgroundColor: bgColor,
        borderRadius: '4px',
        padding: '10px',
        position: 'relative',
        boxShadow: '2px 2px 10px rgba(0,0,0,0.2)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        overflow: 'hidden',
        boxSizing: 'border-box',
      }}
      data-testid="book-cover"
    >
      {/* Decorative pattern */}
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          opacity: 0.1,
        }}
      >
        <pattern
          id={patternId}
          x="0"
          y="0"
          width="20"
          height="20"
          patternUnits="userSpaceOnUse"
        >
          <circle cx="10" cy="10" r="1" fill={textColor} />
        </pattern>
        <rect width="100%" height="100%" fill={`url(#${patternId})`} />
      </svg>

      {/* Content */}
      <div style={{ 
        position: 'relative', 
        zIndex: 1, 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center'
      }}>
        <div
          style={{
            fontSize: Math.min(width * 0.10, 18),
            color: textColor,
            fontWeight: 'bold',
            marginBottom: '8px',
            fontFamily: 'InterVariable, Inter, sans-serif',
            lineHeight: 1.2,
            wordWrap: 'break-word',
            maxWidth: '90%'
          }}
        >
          {truncatedTitle}
        </div>
        {author && (
          <div
            style={{
              fontSize: Math.min(width * 0.06, 12),
              color: textColor,
              opacity: 0.8,
              fontFamily: 'InterVariable, Inter, sans-serif',
              wordWrap: 'break-word',
              maxWidth: '90%'
            }}
          >
            by {truncatedAuthor}
          </div>
        )}
      </div>

      {/* Only render Image if we have a valid imageUrl */}
      {imageUrl && !isPlaceholderImage(imageUrl) && (
        <Image
          src={imageUrl}
          alt={title}
          style={{ 
            objectFit: 'cover',
            width: '100%',
            height: '100%'
          }}
          fill={true}
          sizes="200px"
        />
      )}
    </div>
  );
};

export const isPlaceholderImage = (url: string): boolean => {
  return url.includes('placehold.co') || url.includes('placeholder') || !url;
}; 