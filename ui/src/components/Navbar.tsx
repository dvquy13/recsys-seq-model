"use client"

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { SearchIcon, Loader2 } from 'lucide-react'
import { Container, Wrapper } from '@/components/ui/container'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useDebounce } from '@/lib/hooks/use-debounce'

interface SearchResult {
  id: string
  title: string
  parent_asin?: string
  [key: string]: any
}

export function Navbar() {
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const debouncedSearchQuery = useDebounce(searchQuery, 300)

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setSearchOpen((open) => !open)
      }
    }
    document.addEventListener('keydown', down)
    return () => document.removeEventListener('keydown', down)
  }, [])

  useEffect(() => {
    async function fetchSearchResults() {
      if (!debouncedSearchQuery.trim()) {
        setSearchResults([])
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
        console.log('Search results:', data)
        
        // The API now sets id to parent_asin, so we can use it directly
        setSearchResults(data.items || [])
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
    // Use the item's ID (which is already set to parent_asin by the API)
    if (!item.id) return '#'
    return `/books/${encodeURIComponent(String(item.id))}`
  }

  return (
    <nav className="py-4">
      <Container>
        <Wrapper className="flex items-center justify-between">
          <Link href="/" className="text-lg font-semibold hover:text-primary">
            Home
          </Link>
          
          <div className="relative w-full max-w-sm mx-4">
            <div className="relative flex items-center">
              <SearchIcon className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search books... (⌘K)"
                className="w-full pl-8 rounded-md border border-input"
                onClick={() => setSearchOpen(true)}
              />
            </div>
          </div>
        </Wrapper>
      </Container>
      
      <Dialog open={searchOpen} onOpenChange={setSearchOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <div className="grid gap-4 py-4">
            <div className="relative">
              <SearchIcon className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search books..."
                className="w-full pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <div className="max-h-[300px] overflow-y-auto">
              {isLoading ? (
                <div className="flex items-center justify-center py-6">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  <span>Searching...</span>
                </div>
              ) : searchResults.length === 0 ? (
                <div className="text-center py-6 text-sm text-muted-foreground">
                  {searchQuery.trim() ? 'No results found.' : 'Type to search'}
                </div>
              ) : (
                <div className="grid gap-2">
                  {searchResults.map((item, index) => (
                    <Link
                      key={`${item.id}-${index}`}
                      href={getBookUrl(item)}
                      className="block p-2 rounded-md hover:bg-accent"
                      onClick={() => setSearchOpen(false)}
                    >
                      {item.title || 'Untitled'}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </nav>
  )
} 