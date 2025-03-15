export interface BookRecommendation {
  score: number;
  main_category: string;
  title: string;
  average_rating: number;
  rating_number: number;
  price: string | null;  // Using string since "None" comes as string from backend
  subtitle: string;
  image_url: string;
  parent_asin: string;
}

export interface RecommendationContext {
  user_ids_raw: string[];
  item_seq_raw: string[][];
  candidate_items_raw: string[];
}

export interface RecommendationMetadata {
  rec_id: string;
}

export interface RecommendationResponse {
  recommendations: BookRecommendation[];
  ctx: RecommendationContext;
  metadata: RecommendationMetadata;
} 