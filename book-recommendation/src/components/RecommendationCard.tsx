// components/RecommendationCard.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { BookCover, isPlaceholderImage } from './BookCover';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
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
    <Card
      onClick={handleCardClick}
      className={cn(
        "w-[200px] min-h-[340px] cursor-pointer transition-all duration-200",
        "hover:translate-y-[-5px] hover:shadow-lg"
      )}
    >
      <CardContent className="p-4 flex flex-col items-center">
        <div className="w-full aspect-[2/3] relative mb-4">
          {isPlaceholderImage(image_url) ? (
            <BookCover 
              title={title} 
              width={200} 
              height={300} 
              imageUrl={image_url || ''}
            />
          ) : (
            <div className="relative w-full h-full overflow-hidden rounded-md">
              <Image
                src={image_url || '/placeholder.jpg'}
                alt={title}
                className="object-cover"
                fill={true}
                sizes="200px"
              />
            </div>
          )}
        </div>

        <div className="w-full space-y-2 text-center">
          <h3 className="font-semibold line-clamp-2">{title}</h3>
          
          {average_rating && (
            <div className="flex items-center justify-center gap-2">
              <Badge variant="secondary" className="text-xs">
                {average_rating.toFixed(1)} ★
              </Badge>
              <span className="text-xs text-muted-foreground">
                ({rating_number})
              </span>
            </div>
          )}
          
          {price && price !== "None" && (
            <Badge variant="outline" className="text-xs">
              ${price}
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default RecommendationCard;