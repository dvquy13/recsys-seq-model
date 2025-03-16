'use client'

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Container, Wrapper } from "@/components/ui/container"
import { RecommendationsGrid } from "@/components/RecommendationsGrid"
import { RecentlyViewedGrid } from "@/components/RecentlyViewedGrid"
import { useAppState } from "@/providers/app-state-provider"

// Move this to an environment variable or configuration file
const BOOK_COVER_IMPLEMENTATION = 'textBased'

export default function Home() {
  // Use the app state context for all state management
  const { 
    userId, 
    setUserId, 
    submitUserId,
    recommendations, 
    isLoadingRecommendations,
    recommendationsError: error,
    personalizedRecs
  } = useAppState()
  
  // Add this state to control when to show loading state
  const [isInitialLoad, setIsInitialLoad] = useState(true)

  // After hydration is complete, set isInitialLoad to false
  useEffect(() => {
    setIsInitialLoad(false)
  }, [])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    // Use the submitUserId function from context
    submitUserId(userId)
  }

  // Use personalized recommendations if available, otherwise use the standard ones
  const displayRecommendations = personalizedRecs && personalizedRecs.recommendations && personalizedRecs.recommendations.length > 0
    ? personalizedRecs 
    : recommendations;

  // Determine if we're showing personalized recommendations
  const isPersonalized = !!(personalizedRecs && personalizedRecs.recommendations && personalizedRecs.recommendations.length > 0);

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
              <Button type="submit" disabled={!isInitialLoad && isLoadingRecommendations}>
                {!isInitialLoad && isLoadingRecommendations ? 'Loading...' : 'Submit'}
              </Button>
            </form>

            {error instanceof Error && (
              <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
                {error.message}
              </div>
            )}

            {/* Display Recently Viewed Grid */}
            <RecentlyViewedGrid />

            {displayRecommendations && (
              <RecommendationsGrid
                title={isPersonalized ? "Recommendations Based on Your Browsing" : "Recommendations"}
                recommendations={displayRecommendations.recommendations}
              />
            )}
          </CardContent>
        </Card>
      </Wrapper>
    </Container>
  )
}
