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
    debug: boolean = false
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
          item_seq_raw: [[]],
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