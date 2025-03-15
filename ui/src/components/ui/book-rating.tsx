import { Star } from 'lucide-react';

interface BookRatingProps {
  rating: number;
  ratingCount: number;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  maxStars?: number;
}

export function BookRating({ 
  rating, 
  ratingCount, 
  size = 'md',
  className = '',
  maxStars = 5
}: BookRatingProps) {
  // Size configurations
  const config = {
    sm: {
      starSize: 'w-3 h-3',
      textSize: 'text-xs',
      spacing: 'ml-1',
      gap: 'gap-0.5'
    },
    md: {
      starSize: 'w-4 h-4',
      textSize: 'text-sm',
      spacing: 'ml-1.5',
      gap: 'gap-1'
    },
    lg: {
      starSize: 'w-5 h-5',
      textSize: 'text-base font-semibold',
      spacing: 'ml-2',
      gap: 'gap-1'
    },
  };

  const { starSize, textSize, spacing, gap } = config[size];

  // Create array for partial fill steps
  const fillSteps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
  
  // Primary color for stars
  const primaryColor = '#0057a3';

  // Create an array of stars with more granular fill
  const stars = Array.from({ length: maxStars }, (_, i) => {
    const starValue = i + 1;
    
    // More granular fill calculation:
    // - Full star: rating >= starValue
    // - Partial star: starValue - 1 < rating < starValue
    // - Empty star: rating <= starValue - 1
    const isFull = rating >= starValue;
    const isEmpty = rating <= starValue - 1;
    const partialFill = !isFull && !isEmpty;
    
    // Calculate the fill percentage for partial stars
    let fillPercentage = 0;
    if (partialFill) {
      // Calculate raw percentage
      const rawPercentage = (rating - (starValue - 1)) * 100;
      // Round to nearest 10%
      fillPercentage = Math.round(rawPercentage / 10) * 10;
    }
    
    return (
      <Star 
        key={i}
        className={`${starSize} ${isFull ? 'text-primary' : partialFill ? 'text-primary' : 'text-gray-300'}`}
        fill={isFull ? primaryColor : partialFill ? `url(#partial-${fillPercentage})` : 'none'}
        stroke={isFull || partialFill ? primaryColor : 'currentColor'}
      />
    );
  });

  return (
    <div className={`flex flex-col ${className}`}>
      <div className={`flex items-center ${gap}`}>
        {stars}
        <span className={`${textSize} ${spacing}`}>
          {rating.toFixed(1)}
        </span>
        <span className={`text-muted-foreground ${textSize}`}>
          ({ratingCount.toLocaleString()} ratings)
        </span>
      </div>
      
      {/* SVG defs for partial fills */}
      <svg width="0" height="0" style={{ position: 'absolute' }}>
        <defs>
          {fillSteps.map(percentage => (
            <linearGradient key={percentage} id={`partial-${percentage}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset={`${percentage}%`} stopColor={primaryColor} stopOpacity="1" />
              <stop offset={`${percentage}%`} stopColor={primaryColor} stopOpacity="0" />
            </linearGradient>
          ))}
        </defs>
      </svg>
    </div>
  );
} 