'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import type { Recommendation, RecommendationsResponse } from '@/types/api'
import { 
  getRecentlyViewedBooks, 
  addRecentlyViewedBook, 
  clearRecentlyViewedBooks, 
  type RecentlyViewedBook,
  RECENTLY_VIEWED_CHANGE_EVENT 
} from '@/lib/recentlyViewed'
import {
  updatePersonalizedRecs,
  getCachedPersonalizedRecs,
  clearPersonalizedRecs,
  PERSONALIZED_RECS_UPDATED_EVENT
} from '@/lib/personalizedRecs'
import { MAX_DISPLAY_ITEMS } from '@/lib/config/recentlyViewed'
import { recommendationsApi } from '@/lib/api'

// Keys for local storage
const USER_ID_STORAGE_KEY = 'last-submitted-user-id'

// Types for our context
interface AppStateContextType {
  // User state
  userId: string
  setUserId: (id: string) => void
  submittedUserId: string | null
  submitUserId: (id: string) => void
  
  // Recently viewed books
  recentlyViewedBooks: Recommendation[]
  addBookToRecentlyViewed: (book: Recommendation) => void
  clearRecentlyViewed: () => void
  
  // Personalized recommendations
  personalizedRecs: RecommendationsResponse | null
  updatePersonalizedRecommendations: (userId?: string) => Promise<boolean>
  clearPersonalizedRecommendations: () => void
  
  // Standard recommendations
  recommendations: RecommendationsResponse | null
  isLoadingRecommendations: boolean
  recommendationsError: Error | null
}

// Default values for the context
const defaultContext: AppStateContextType = {
  userId: '',
  setUserId: () => {},
  submittedUserId: null,
  submitUserId: () => {},
  
  recentlyViewedBooks: [],
  addBookToRecentlyViewed: () => {},
  clearRecentlyViewed: () => {},
  
  personalizedRecs: null,
  updatePersonalizedRecommendations: async () => false,
  clearPersonalizedRecommendations: () => {},
  
  recommendations: null,
  isLoadingRecommendations: false,
  recommendationsError: null
}

// Create the context
const AppStateContext = createContext<AppStateContextType>(defaultContext)

// Provider component
export function AppStateProvider({ children }: { children: ReactNode }) {
  // User state
  const [userId, setUserId] = useState('')
  const [submittedUserId, setSubmittedUserId] = useState<string | null>(() => {
    // Initialize from localStorage if we're in the browser
    if (typeof window !== 'undefined') {
      return localStorage.getItem(USER_ID_STORAGE_KEY)
    }
    return null
  })
  
  // Recently viewed books state
  const [recentlyViewedBooks, setRecentlyViewedBooks] = useState<Recommendation[]>([])
  
  // Personalized recommendations state
  const [personalizedRecs, setPersonalizedRecs] = useState<RecommendationsResponse | null>(null)
  
  // Standard recommendations state with React Query-like interface
  const [recommendations, setRecommendations] = useState<RecommendationsResponse | null>(null)
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false)
  const [recommendationsError, setRecommendationsError] = useState<Error | null>(null)
  
  // Load recently viewed books
  const loadRecentlyViewedBooks = () => {
    // Get recently viewed books and sort by viewedAt (most recent first)
    const books = getRecentlyViewedBooks()
      .sort((a, b) => b.viewedAt - a.viewedAt)
    
    // Convert RecentlyViewedBook[] to Recommendation[] and ensure all required fields exist
    const recommendations: Recommendation[] = books
      .filter(book => {
        // Filter out any books with missing required properties
        return book && 
          typeof book.title === 'string' && 
          typeof book.parent_asin === 'string' &&
          typeof book.average_rating === 'number' && 
          typeof book.rating_number === 'number'
      })
      .map(({ viewedAt, ...book }) => ({
        ...book,
        // Ensure all required properties are valid
        average_rating: book.average_rating || 0,
        rating_number: book.rating_number || 0,
        price: book.price || null,
        subtitle: book.subtitle || '',
        main_category: book.main_category || 'Unknown',
        score: book.score || 0
      }))
    
    // Only take the first MAX_DISPLAY_ITEMS
    const newRecentlyViewed = recommendations.slice(0, MAX_DISPLAY_ITEMS);
    
    // Only update state if the books have actually changed
    if (JSON.stringify(newRecentlyViewed) !== JSON.stringify(recentlyViewedBooks)) {
      setRecentlyViewedBooks(newRecentlyViewed);
    }
  }
  
  // Add a book to recently viewed
  const addBookToRecentlyViewed = (book: Recommendation) => {
    // Make sure the book has all required properties before saving
    if (book && typeof book.title === 'string' && typeof book.parent_asin === 'string') {
      // Ensure all required properties exist with defaults if needed
      const validBook: Recommendation = {
        ...book,
        average_rating: typeof book.average_rating === 'number' ? book.average_rating : 0,
        rating_number: typeof book.rating_number === 'number' ? book.rating_number : 0,
        price: book.price || null,
        subtitle: book.subtitle || '',
        main_category: book.main_category || 'Unknown',
        score: typeof book.score === 'number' ? book.score : 0,
        image_url: book.image_url || ''
      }
      
      // Check if the book is already in the recently viewed list to avoid unnecessary updates
      const existingBook = recentlyViewedBooks.find(b => b.parent_asin === book.parent_asin);
      if (existingBook) {
        // If the book is already first in the list (most recently viewed), don't do anything
        if (recentlyViewedBooks[0]?.parent_asin === book.parent_asin) {
          return;
        }
      }
      
      // Add the validated book to recently viewed
      addRecentlyViewedBook(validBook);
      
      // Update personalized recommendations
      // Use a setTimeout to avoid triggering an immediate state update
      setTimeout(() => {
        updatePersonalizedRecommendations().catch(error => {
          console.error('Failed to update personalized recommendations:', error);
        });
      }, 0);
    }
  }
  
  // Clear recently viewed books
  const clearRecentlyViewed = () => {
    clearRecentlyViewedBooks()
    setRecentlyViewedBooks([])
  }
  
  // Submit user ID
  const submitUserId = (id: string) => {
    // If the user ID has changed, clear the recently viewed books and personalized recs
    if (id !== submittedUserId) {
      clearRecentlyViewed()
      clearPersonalizedRecommendations()
    }
    
    localStorage.setItem(USER_ID_STORAGE_KEY, id)
    setSubmittedUserId(id)
    
    // Fetch recommendations for the new user ID
    fetchRecommendations(id)
  }
  
  // Update personalized recommendations
  const updatePersonalizedRecommendations = async (userIdParam?: string) => {
    const effectiveUserId = userIdParam || submittedUserId || 'guest'
    const result = await updatePersonalizedRecs(effectiveUserId)
    
    // Update local state
    setPersonalizedRecs(getCachedPersonalizedRecs())
    
    return result
  }
  
  // Clear personalized recommendations
  const clearPersonalizedRecommendations = () => {
    clearPersonalizedRecs()
    setPersonalizedRecs(null)
  }
  
  // Fetch recommendations
  const fetchRecommendations = async (userId: string) => {
    setIsLoadingRecommendations(true)
    setRecommendationsError(null)
    
    try {
      const data = await recommendationsApi.getRecommendations(userId || '')
      setRecommendations(data)
    } catch (error) {
      if (error instanceof Error) {
        setRecommendationsError(error)
      } else {
        setRecommendationsError(new Error('Failed to fetch recommendations'))
      }
    } finally {
      setIsLoadingRecommendations(false)
    }
  }
  
  // Effect for initial loading of recently viewed books
  useEffect(() => {
    loadRecentlyViewedBooks()
    
    // Set up listeners for both storage events and our custom event
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === 'recently-viewed-books' || event.key === null) {
        loadRecentlyViewedBooks()
      }
    }
    
    const handleRecentlyViewedChange = () => {
      loadRecentlyViewedBooks()
    }
    
    // Add event listeners
    window.addEventListener('storage', handleStorageChange)
    window.addEventListener(RECENTLY_VIEWED_CHANGE_EVENT, handleRecentlyViewedChange)
    
    // Clean up event listeners on unmount
    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener(RECENTLY_VIEWED_CHANGE_EVENT, handleRecentlyViewedChange)
    }
  }, [])
  
  // Effect for initial loading of personalized recommendations
  useEffect(() => {
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
  
  // Effect to fetch recommendations when submitted user ID changes
  useEffect(() => {
    if (submittedUserId !== null) {
      fetchRecommendations(submittedUserId)
    }
  }, [submittedUserId])
  
  // Context value
  const value: AppStateContextType = {
    userId,
    setUserId,
    submittedUserId,
    submitUserId,
    
    recentlyViewedBooks,
    addBookToRecentlyViewed,
    clearRecentlyViewed,
    
    personalizedRecs,
    updatePersonalizedRecommendations,
    clearPersonalizedRecommendations,
    
    recommendations,
    isLoadingRecommendations,
    recommendationsError
  }
  
  return (
    <AppStateContext.Provider value={value}>
      {children}
    </AppStateContext.Provider>
  )
}

// Custom hook to use app state
export const useAppState = () => useContext(AppStateContext) 