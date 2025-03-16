import { NextRequest, NextResponse } from 'next/server'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    console.log('Search request:', body)
    
    const response = await fetch(`${API_URL}/items/search_by_title`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }
    
    const data = await response.json()
    console.log('Search API raw response:', data)
    
    // Make sure we have the right structure for our frontend
    const formattedResponse = {
      items: Array.isArray(data.items) ? data.items.map((item: any, index: number) => ({
        // Use parent_asin as the ID for book links
        id: item.parent_asin || `item-${index}`,
        title: item.title || '',
        ...item
      })) : []
    }
    
    console.log('Formatted search response:', formattedResponse)
    return NextResponse.json(formattedResponse)
  } catch (error) {
    console.error('Error in search API route:', error)
    return NextResponse.json(
      { error: 'Failed to search items' },
      { status: 500 }
    )
  }
} 