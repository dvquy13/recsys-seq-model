import { Card } from "@/components/ui/card"
import type { Recommendation } from "@/types/api"
import { BookCover } from "@/components/book-covers"
import { getConfig } from "@/lib/config"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { BookRating } from "@/components/ui/book-rating"
import { BookPrice } from "@/components/ui/book-price"

export interface RecommendationCardProps {
  recommendation: Recommendation;
  showScore?: boolean;
  linkPrefix?: string;
  className?: string;
}

export function RecommendationCard({ 
  recommendation, 
  showScore = true,
  linkPrefix = '/books',
  className = 'p-4 hover:bg-accent transition-colors'
}: RecommendationCardProps) {
  const bookCoverImplementation = getConfig('ui', 'bookCoverImplementation')
  
  return (
    <Link href={`${linkPrefix}/${recommendation.parent_asin}`}>
      <Card className={className}>
        <div className="flex gap-4">
          <div className="flex-shrink-0" style={{ width: 96, height: 144 }}>
            <BookCover
              title={recommendation.title}
              imageUrl={recommendation.image_url}
              width={96}
              height={144}
              parent_asin={recommendation.parent_asin}
              implementation={bookCoverImplementation}
            />
          </div>
          <div className="flex-grow min-w-0 flex flex-col">
            <div>
              <h4 
                className="font-semibold line-clamp-2" 
                title={recommendation.title}
              >
                {recommendation.title}
              </h4>
              {recommendation.subtitle && (
                <p 
                  className="text-sm text-muted-foreground truncate" 
                  title={recommendation.subtitle}
                >
                  {recommendation.subtitle}
                </p>
              )}
            </div>
            
            <div className="mt-2 space-y-2.5">
              <div className="flex flex-wrap items-center">
                <BookRating 
                  rating={recommendation.average_rating} 
                  ratingCount={recommendation.rating_number}
                  size="sm"
                  className="flex-shrink-0 w-full"
                />
              </div>
              
              <div className="flex items-center gap-2">
                <BookPrice 
                  price={recommendation.price} 
                  size="sm"
                />
                
                {showScore && (
                  <Badge variant="secondary" className="text-xs ml-auto">
                    Score: {recommendation.score.toFixed(2)}
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </div>
      </Card>
    </Link>
  )
} 