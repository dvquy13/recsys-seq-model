export interface Recommendation {
  score: number;
  main_category: string;
  title: string;
  average_rating: number;
  rating_number: number;
  price: string | null;
  subtitle: string;
  image_url: string;
  parent_asin: string;
}

export interface ApiContext {
  user_ids_raw: string[];
  item_seq_raw: string[][];
  candidate_items_raw: string[];
}

export interface ApiMetadata {
  rec_id: string;
}

export interface RecommendationsResponse {
  recommendations: Recommendation[];
  ctx: ApiContext;
  metadata: ApiMetadata;
}

export interface RecommendationsRequest {
  user_ids_raw: string[];
  item_seq_raw: string[][];
  candidate_items_raw: string[];
} 