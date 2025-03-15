'use client'

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { RecommendationsResponse } from "@/types/api"
import { BookCover } from "@/components/book-covers"
import { recommendationsApi } from "@/lib/api"

// Move this to an environment variable or configuration file
const BOOK_COVER_IMPLEMENTATION = 'textBased' // or 'dicebear'

interface RecommendationCardProps {
  recommendation: RecommendationsResponse['recommendations'][0]
}

const RecommendationCard = ({ recommendation }: RecommendationCardProps) => (
  <Card className="p-4">
    <div className="flex gap-4">
      <BookCover
        title={recommendation.title}
        imageUrl={recommendation.image_url}
        width={96}
        height={144}
        parent_asin={recommendation.parent_asin}
        implementation={BOOK_COVER_IMPLEMENTATION}
      />
      <div>
        <h4 className="font-semibold">{recommendation.title}</h4>
        <p className="text-sm text-gray-500">{recommendation.subtitle}</p>
        <div className="mt-2 text-sm">
          <p>Rating: {recommendation.average_rating} ({recommendation.rating_number} reviews)</p>
          {recommendation.price && <p>Price: ${recommendation.price}</p>}
          <p>Score: {recommendation.score.toFixed(2)}</p>
        </div>
      </div>
    </div>
  </Card>
)

export default function Home() {
  const [userId, setUserId] = useState("")
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<RecommendationsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const data = await recommendationsApi.getRecommendations(userId)
      setRecommendations(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch recommendations')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto py-10">
      <Card>
        <CardHeader>
          <CardTitle>User Lookup</CardTitle>
          <CardDescription>Enter a user ID to fetch their recommendations</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="userId">User ID</Label>
              <Input
                id="userId"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Enter user ID"
                required
              />
            </div>
            <Button type="submit" disabled={loading}>
              {loading ? 'Loading...' : 'Submit'}
            </Button>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
              {error}
            </div>
          )}

          {recommendations && (
            <div className="mt-6 space-y-4">
              <h3 className="text-lg font-semibold">Recommendations</h3>
              <div className="grid gap-4">
                {recommendations.recommendations.map((rec, index) => (
                  <RecommendationCard key={index} recommendation={rec} />
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
