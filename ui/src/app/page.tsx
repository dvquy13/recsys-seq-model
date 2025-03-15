'use client'

import { useState, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { RecommendationsResponse } from "@/types/api"
import { recommendationsApi } from "@/lib/api"
import { RecommendationCard } from "@/components/RecommendationCard"
import { Container, Wrapper } from "@/components/ui/container"

// Move this to an environment variable or configuration file
const BOOK_COVER_IMPLEMENTATION = 'textBased'

const STORAGE_KEY = 'last-submitted-user-id'

interface RecommendationCardProps {
  recommendation: RecommendationsResponse['recommendations'][0]
}

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

            {recommendations && (
              <div className="mt-6 space-y-4">
                <h3 className="text-lg font-semibold">Recommendations</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {recommendations.recommendations.map((rec, index) => (
                    <RecommendationCard key={index} recommendation={rec} />
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </Wrapper>
    </Container>
  )
}
