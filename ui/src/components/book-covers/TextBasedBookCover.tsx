'use client';

import React from 'react';
import Image from 'next/image';
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { BookCoverProps } from './types';

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

const getPatternId = (parent_asin: string | undefined, title: string) => {
  // Use parent_asin if available, otherwise use title
  const stringToHash = parent_asin || title;
  
  // Create a simple hash from the string
  const hash = stringToHash.split('').reduce((acc, char) => {
    return ((acc << 5) - acc) + char.charCodeAt(0);
  }, 0);
  
  // Use the absolute value of hash to ensure positive number
  const positiveHash = Math.abs(hash);
  return `pattern-${positiveHash}`;
};

export const TextBasedBookCover: React.FC<BookCoverProps> = ({
  title,
  author,
  width = 200,
  height = 300,
  backgroundColor,
  textColor = '#FFFFFF',
  imageUrl,
  parent_asin,
}) => {
  const bgColor = backgroundColor || getColorFromAsin(parent_asin, title);
  const truncatedTitle = truncateText(title, 50);
  const truncatedAuthor = author ? truncateText(author, 30) : '';
  const patternId = getPatternId(parent_asin, title);
  const hasValidImage = imageUrl && !isPlaceholderImage(imageUrl);

  return (
    <Card
      className={cn(
        "relative overflow-hidden transition-shadow hover:shadow-lg",
        hasValidImage ? "" : "flex flex-col justify-center items-center"
      )}
      style={{
        width,
        height,
        backgroundColor: bgColor,
      }}
      data-testid="book-cover"
    >
      {/* Decorative pattern for non-image covers */}
      {!hasValidImage && (
        <svg
          className="absolute inset-0 w-full h-full opacity-10"
          aria-hidden="true"
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
      )}

      {/* Content */}
      {!hasValidImage && (
        <div className="relative z-10 p-4 text-center">
          <h3
            className="text-base font-bold leading-tight mb-2"
            style={{
              color: textColor,
              fontSize: Math.min(width * 0.12, 20),
              maxWidth: '90%'
            }}
          >
            {truncatedTitle}
          </h3>
          {author && (
            <p
              className="text-sm opacity-80"
              style={{
                color: textColor,
                fontSize: Math.min(width * 0.06, 12),
                maxWidth: '90%'
              }}
            >
              by {truncatedAuthor}
            </p>
          )}
        </div>
      )}

      {/* Book cover image */}
      {hasValidImage && (
        <Image
          src={imageUrl}
          alt={title}
          className="object-cover w-full h-full"
          fill
          sizes={`${width}px`}
        />
      )}
    </Card>
  );
};

export const isPlaceholderImage = (url: string): boolean => {
  return url.includes('placehold.co') || url.includes('placeholder') || !url;
}; 