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
  const [imageError, setImageError] = React.useState(false);

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
      const imageUrl = isPlaceholderImage(image_url) || imageError
        ? '' 
        : image_url;

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
      
      const encodedData = encodeURIComponent(JSON.stringify(validatedRecommendation));
      router.push(`/book-details?data=${encodedData}`);
    } catch (error) {
      console.error('Error preparing book data:', error);
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
        "w-[180px] flex-shrink-0 cursor-pointer",
        "transition-all duration-200 hover:shadow-lg",
        "bg-card text-card-foreground"
      )}
    >
      <CardContent className="p-3">
        <div className="aspect-[2/3] relative mb-3">
          {isPlaceholderImage(image_url) || imageError ? (
            <BookCover 
              title={title} 
              width={180} 
              height={270} 
              imageUrl={image_url || ''}
            />
          ) : (
            <div className="relative w-full h-full overflow-hidden rounded-md">
              <Image
                src={image_url}
                alt={title}
                className="object-cover"
                fill={true}
                sizes="180px"
                onError={() => setImageError(true)}
                priority={true}
              />
            </div>
          )}
        </div>

        <div className="space-y-1.5">
          <h3 className="font-medium text-sm line-clamp-2 min-h-[2.5rem]">
            {title}
          </h3>
          
          <div className="flex items-center gap-1.5 flex-wrap">
            {average_rating > 0 && (
              <Badge variant="secondary" className="text-xs px-1.5 py-0">
                {average_rating.toFixed(1)} ★
              </Badge>
            )}
            {rating_number > 0 && (
              <span className="text-xs text-muted-foreground">
                ({rating_number.toLocaleString()})
              </span>
            )}
          </div>
          
          {price && price !== "None" && (
            <div className="mt-1">
              <Badge variant="outline" className="text-xs px-1.5 py-0">
                ${price}
              </Badge>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default RecommendationCard;