'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: Infinity, // Keep data fresh forever until explicitly invalidated
        gcTime: 24 * 60 * 60 * 1000, // Keep unused data in cache for 24 hours
        refetchOnMount: false, // Don't refetch on mount
        refetchOnWindowFocus: false, // Don't refetch on window focus
        retry: false, // Don't retry failed requests automatically
      },
    },
  }))

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
} 