'use client'

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { RecommendationsResponse } from "@/types/api"
import { generateBookCover, shouldGenerateBookCover } from "@/lib/bookCover"

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
      const response = await fetch('http://localhost:8000/recs/retrieve?count=10&debug=false', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'accept': 'application/json',
        },
        body: JSON.stringify({
          user_ids_raw: [userId],
          item_seq_raw: [[]],
          candidate_items_raw: []
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: RecommendationsResponse = await response.json()
      setRecommendations(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch recommendations')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>User Lookup</CardTitle>
          <CardDescription>Enter a user ID to fetch their recommendations</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
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
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? 'Loading...' : 'Submit'}
              </Button>
            </div>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-md">
              {error}
            </div>
          )}

          {recommendations && (
            <div className="mt-6 space-y-4">
              <h3 className="text-lg font-semibold">Recommendations</h3>
              <div className="grid gap-4">
                {recommendations.recommendations.map((rec, index) => (
                  <Card key={index} className="p-4">
                    <div className="flex gap-4">
                      <img 
                        src={shouldGenerateBookCover(rec.image_url) ? generateBookCover(rec.title) : rec.image_url} 
                        alt={rec.title}
                        className="w-24 h-36 object-cover rounded-md"
                      />
                      <div>
                        <h4 className="font-semibold">{rec.title}</h4>
                        <p className="text-sm text-gray-500">{rec.subtitle}</p>
                        <div className="mt-2 text-sm">
                          <p>Rating: {rec.average_rating} ({rec.rating_number} reviews)</p>
                          {rec.price && <p>Price: ${rec.price}</p>}
                          <p>Score: {rec.score.toFixed(2)}</p>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
