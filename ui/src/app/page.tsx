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
import { clearRecentlyViewedBooks } from "@/lib/recentlyViewed"
import { 
  getCachedPersonalizedRecs, 
  PERSONALIZED_RECS_UPDATED_EVENT,
  clearPersonalizedRecs
} from "@/lib/personalizedRecs"

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
  const [personalizedRecs, setPersonalizedRecs] = useState<RecommendationsResponse | null>(null)
  // Add this state to control when to show loading state
  const [isInitialLoad, setIsInitialLoad] = useState(true)

  // Standard recommendations from API
  const { data: recommendations, isLoading, error } = useQuery({
    queryKey: ['recommendations', submittedUserId],
    queryFn: () => recommendationsApi.getRecommendations(submittedUserId || ''),
    enabled: submittedUserId !== null,
  })

  // After hydration is complete, set isInitialLoad to false
  useEffect(() => {
    setIsInitialLoad(false)
  }, [])

  // Load personalized recommendations initially and when updated
  useEffect(() => {
    // Load initially
    setPersonalizedRecs(getCachedPersonalizedRecs())
    
    // Set up listener for personalized recommendations updates
    const handlePersonalizedRecsUpdated = () => {
      setPersonalizedRecs(getCachedPersonalizedRecs())
    }
    
    window.addEventListener(PERSONALIZED_RECS_UPDATED_EVENT, handlePersonalizedRecsUpdated)
    
    return () => {
      window.removeEventListener(PERSONALIZED_RECS_UPDATED_EVENT, handlePersonalizedRecsUpdated)
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    
    // If the user ID has changed, clear the recently viewed books and personalized recs
    if (userId !== submittedUserId) {
      clearRecentlyViewedBooks();
      clearPersonalizedRecs();
    }
    
    localStorage.setItem(STORAGE_KEY, userId)
    setSubmittedUserId(userId)
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
              <Button type="submit" disabled={!isInitialLoad && isLoading}>
                {!isInitialLoad && isLoading ? 'Loading...' : 'Submit'}
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
