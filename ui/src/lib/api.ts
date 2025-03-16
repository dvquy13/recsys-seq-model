import type { RecommendationsRequest, RecommendationsResponse } from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export const recommendationsApi = {
  async getRecommendations(
    userId: string,
    count: number = 10,
    debug: boolean = false,
    itemSeqRaw: string[][] = [[]]
  ): Promise<RecommendationsResponse> {
    const response = await fetch(
      `${API_BASE_URL}/recs/retrieve?count=${count}&debug=${debug}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'accept': 'application/json',
        },
        body: JSON.stringify({
          user_ids_raw: [userId],
          item_seq_raw: itemSeqRaw,
          candidate_items_raw: []
        } as RecommendationsRequest)
      }
    );

    if (!response.ok) {
      throw new ApiError(response.status, `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }
};

export async function getBookDetails(bookId: string) {
  try {
    console.log(`Fetching book details for ID: ${bookId}`);
    
    const normalizedId = bookId.trim();
    if (!normalizedId) {
      console.error('Invalid book ID: empty');
      return null;
    }
    
    const response = await fetch(`${API_BASE_URL}/items/get_by_ids?debug=false`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ item_ids: [normalizedId], "debug": false }),
    });

    if (!response.ok) {
      console.error(`API error: ${response.status}`);
      return null;
    }

    const data = await response.json();
    console.log('Book details response:', data);
    
    if (!data.items || !Array.isArray(data.items) || data.items.length === 0) {
      console.error('No book found with ID:', normalizedId);
      return null;
    }
    
    const book = data.items[0];
    
    // Ensure the book has the required fields
    if (!book || typeof book !== 'object') {
      console.error('Invalid book data structure:', book);
      return null;
    }
    
    // Provide defaults for critical fields to prevent UI errors
    return {
      id: book.parent_asin || normalizedId,
      parent_asin: book.parent_asin || normalizedId,
      title: book.title || 'Untitled Book',
      image_url: book.image_url || '',
      average_rating: book.average_rating || 0,
      rating_number: book.rating_number || 0,
      price: book.price !== undefined ? book.price : 0,
      main_category: book.main_category || 'Uncategorized',
      ...book
    };
  } catch (error) {
    console.error('Error fetching book details:', error);
    return null;
  }
} 