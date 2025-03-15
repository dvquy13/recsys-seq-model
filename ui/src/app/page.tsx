'use client'

import { useState, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { RecommendationsResponse } from "@/types/api"
import { recommendationsApi } from "@/lib/api"
import { Container, Wrapper } from "@/components/ui/container"
import { RecommendationsGrid } from "@/components/RecommendationsGrid"
import { RecentlyViewedGrid } from "@/components/RecentlyViewedGrid"

// Move this to an environment variable or configuration file
const BOOK_COVER_IMPLEMENTATION = 'textBased'

const STORAGE_KEY = 'last-submitted-user-id'

export default function Home() {
  const [userId, setUserId] = useState("")
  const [submittedUserId, setSubmittedUserId] = useState<string | null>(() => {
    // Initialize from localStorage if we're in the browser
    if (typeof window !== 'undefined') {
      return localStorage.getItem(STORAGE_KEY)
    }
    return null
  })

  const { data: recommendations, isLoading, error } = useQuery({
    queryKey: ['recommendations', submittedUserId],
    queryFn: () => recommendationsApi.getRecommendations(submittedUserId || ''),
    enabled: submittedUserId !== null,
  })

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    localStorage.setItem(STORAGE_KEY, userId)
    setSubmittedUserId(userId)
  }

  return (
    <Container className="py-6">
      <Wrapper className="p-0">
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
                  placeholder="Enter user ID (optional)"
                />
              </div>
              <Button type="submit" disabled={isLoading}>
                {isLoading ? 'Loading...' : 'Submit'}
              </Button>
            </form>

            {error instanceof Error && (
              <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
                {error.message}
              </div>
            )}

            {/* Display Recently Viewed Grid */}
            <RecentlyViewedGrid />

            {recommendations && (
              <RecommendationsGrid
                title="Recommendations"
                recommendations={recommendations.recommendations}
              />
            )}
          </CardContent>
        </Card>
      </Wrapper>
    </Container>
  )
}
