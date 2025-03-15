'use client';

import React from 'react';

interface BookCoverProps {
  title: string;
  author?: string;
  width?: number;
  height?: number;
  backgroundColor?: string;
  textColor?: string;
}

const getRandomColor = () => {
  const colors = [
    '#2D3047', // Dark blue
    '#419D78', // Green
    '#E0A458', // Orange
    '#8D5B4C', // Brown
    '#562C2C', // Dark red
    '#2C365E', // Navy
    '#484D6D', // Slate
  ] as const;
  return colors[Math.floor(Math.random() * colors.length)];
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
}: BookCoverProps) => {
  const bgColor = backgroundColor || getRandomColor();
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

      {/* Bottom decoration - removed to match image layout */}
    </div>
  );
};

export const isPlaceholderImage = (url: string): boolean => {
  return url.includes('placehold.co') || url.includes('placeholder') || !url;
}; 