import { Card } from "@/components/ui/card"
import type { Recommendation } from "@/types/api"
import { BookCover } from "@/components/book-covers"
import { getConfig } from "@/lib/config"
import Link from "next/link"

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
          <div className="flex-grow min-w-0">
            <h4 className="font-semibold">{recommendation.title}</h4>
            <p className="text-sm text-gray-500">{recommendation.subtitle}</p>
            <div className="mt-2 text-sm">
              <p>Rating: {recommendation.average_rating} ({recommendation.rating_number} reviews)</p>
              {recommendation.price && <p>Price: ${recommendation.price}</p>}
              {showScore && <p>Score: {recommendation.score.toFixed(2)}</p>}
            </div>
          </div>
        </div>
      </Card>
    </Link>
  )
} 