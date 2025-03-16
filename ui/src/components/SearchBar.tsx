"use client"

import { useState, useEffect, useRef } from 'react'
import { SearchIcon, Loader2 } from 'lucide-react'
import { useDebounce } from '@/lib/hooks/use-debounce'
import { Input } from "@/components/ui/input"
import Link from 'next/link'
import { BookCover } from '@/components/book-covers'
import { getConfig } from '@/lib/config'
import { usePathname } from 'next/navigation'

// TODO: Using Shadcn UI Command component
// Currently I couldn't get it to work, typing something leads to nothing shown up
// Maybe related issue: https://github.com/shadcn-ui/ui/issues/2652

interface SearchResult {
  id: string
  title: string
  parent_asin?: string
  image_url?: string
  author?: string
  [key: string]: any
}

export function SearchBar() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(false)
  const debouncedSearchQuery = useDebounce(searchQuery, 300)
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const bookCoverImplementation = getConfig('ui', 'bookCoverImplementation')
  const pathname = usePathname()
  
  // Reset state when route changes
  useEffect(() => {
    setIsOpen(false)
  }, [pathname])

  // Handle clicks outside to close the results
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    
    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  useEffect(() => {
    async function fetchSearchResults() {
      if (!debouncedSearchQuery.trim()) {
        setSearchResults([])
        setIsOpen(false)
        return
      }
      
      setIsLoading(true)
      
      try {
        const response = await fetch('/api/items/search_by_title', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: debouncedSearchQuery,
            limit: 10,
          }),
        })
        
        if (!response.ok) {
          throw new Error('Failed to fetch search results')
        }
        
        const data = await response.json()
        
        // Check if data.items exists and is an array
        if (data && Array.isArray(data.items)) {
          setSearchResults(data.items)
        } else if (data && Array.isArray(data)) {
          // In case the API returns the array directly
          setSearchResults(data)
        } else {
          console.error('Unexpected API response format:', data)
          setSearchResults([])
        }
      } catch (error) {
        console.error('Error searching items:', error)
        setSearchResults([])
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchSearchResults()
  }, [debouncedSearchQuery])

  // Function to safely get book URL
  const getBookUrl = (item: SearchResult) => {
    if (!item.id) return '#'
    return `/books/${encodeURIComponent(String(item.id))}`
  }

  // Only show dropdown when actually searching or when we have results
  useEffect(() => {
    if (searchQuery) {
      setIsOpen(isLoading || searchResults.length > 0)
    } else {
      setIsOpen(false)
    }
  }, [searchQuery, searchResults, isLoading])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value)
  }

  // Handle focus - only open if we have something to show
  const handleFocus = () => {
    if (searchQuery && (isLoading || searchResults.length > 0)) {
      setIsOpen(true)
    }
  }

  // Function to handle result click
  const handleResultClick = () => {
    setIsOpen(false)
  }

  return (
    <div className="w-full max-w-lg mx-auto my-6" ref={containerRef}>
      <div className="relative">
        {/* Custom search input instead of CommandInput */}
        <div className="relative flex items-center">
          <SearchIcon className="absolute left-3 h-4 w-4 text-muted-foreground" />
          <Input
            ref={inputRef}
            type="text"
            placeholder="Search books..."
            value={searchQuery}
            onChange={handleInputChange}
            onFocus={handleFocus}
            className="pl-9 pr-4 py-2 w-full h-10 rounded-lg border shadow-md"
          />
        </div>
        
        {/* Only show dropdown when we have content to display */}
        {isOpen && (isLoading || searchResults.length > 0) && (
          <div className="absolute z-50 mt-1 w-full bg-white rounded-md border shadow-lg overflow-hidden">
            <div className="max-h-[400px] overflow-y-auto p-1">
              {isLoading && (
                <div className="flex items-center justify-center py-6">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  <span>Searching...</span>
                </div>
              )}
              
              {!isLoading && searchQuery && searchResults.length === 0 && (
                <div className="py-6 text-center text-sm text-gray-500">
                  No results found.
                </div>
              )}
              
              {!isLoading && searchResults.length > 0 && (
                <div className="py-1">
                  <div className="px-2 py-1.5 text-xs font-semibold text-gray-500">
                    Books
                  </div>
                  {searchResults.map((item, index) => (
                    <Link
                      key={`${item.id}-${index}`}
                      href={getBookUrl(item)}
                      className="flex items-center gap-3 px-2 py-2 text-sm rounded-md hover:bg-gray-100 cursor-pointer"
                      onClick={handleResultClick}
                    >
                      {/* Book cover image with custom styling to override rounded corners */}
                      <div className="flex-shrink-0" style={{ width: 30, height: 45 }}>
                        <div className="[&>*]:!rounded-none [&>div]:!rounded-none" style={{ width: 30, height: 45 }}>
                          <BookCover
                            title={item.title}
                            imageUrl={item.image_url}
                            width={30}
                            height={45}
                            parent_asin={item.id || item.parent_asin}
                            implementation={bookCoverImplementation}
                          />
                        </div>
                      </div>
                      <span className="truncate">{item.title || 'Untitled'}</span>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 