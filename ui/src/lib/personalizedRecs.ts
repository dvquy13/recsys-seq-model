import { getRecentlyViewedBooks } from './recentlyViewed';
import { recommendationsApi } from './api';
import type { RecommendationsResponse } from '@/types/api';

// Storage keys for cache
const PERSONALIZED_RECS_CACHE_KEY = 'personalized-recommendations';
const LAST_VIEWED_ITEMS_CACHE_KEY = 'last-viewed-items';

// Custom event for notifying components about new personalized recommendations
export const PERSONALIZED_RECS_UPDATED_EVENT = 'personalized-recs-updated';

// Helper function to dispatch event when personalized recs are updated
function dispatchPersonalizedRecsUpdateEvent() {
  if (typeof window !== 'undefined') {
    const event = new CustomEvent(PERSONALIZED_RECS_UPDATED_EVENT);
    window.dispatchEvent(event);
  }
}

/**
 * Get the cached personalized recommendations
 */
export function getCachedPersonalizedRecs(): RecommendationsResponse | null {
  if (typeof window === 'undefined') {
    return null; // Return null on server-side
  }
  
  try {
    const cachedData = localStorage.getItem(PERSONALIZED_RECS_CACHE_KEY);
    return cachedData ? JSON.parse(cachedData) : null;
  } catch (error) {
    console.error('Failed to parse personalized recommendations from cache', error);
    return null;
  }
}

/**
 * Get the cached last viewed items array (for comparison)
 */
function getCachedLastViewedItems(): string[] {
  if (typeof window === 'undefined') {
    return [];
  }
  
  try {
    const cachedData = localStorage.getItem(LAST_VIEWED_ITEMS_CACHE_KEY);
    return cachedData ? JSON.parse(cachedData) : [];
  } catch (error) {
    console.error('Failed to parse last viewed items from cache', error);
    return [];
  }
}

/**
 * Update personalized recommendations based on recently viewed books
 * Returns true if the recommendations were updated, false otherwise
 */
export async function updatePersonalizedRecs(userId: string): Promise<boolean> {
  if (typeof window === 'undefined') {
    return false; // Do nothing on server-side
  }
  
  try {
    // Get recently viewed books
    const recentlyViewed = getRecentlyViewedBooks();
    
    // Extract the parent_asin values from recently viewed books (ordered by viewedAt ascending)
    const viewedItems = recentlyViewed
      .sort((a, b) => a.viewedAt - b.viewedAt) // Sort by oldest first
      .map(book => book.parent_asin);
    
    // Check if the viewedItems array is the same as the cached one
    const cachedViewedItems = getCachedLastViewedItems();
    const itemsAreEqual = 
      JSON.stringify(viewedItems) === JSON.stringify(cachedViewedItems);
    
    // If there are no changes to viewed items, don't update
    if (itemsAreEqual && viewedItems.length > 0) {
      return false;
    }
    
    // Call the API with the updated item_seq_raw
    const response = await recommendationsApi.getRecommendations(
      userId,
      10,  // Default count
      false, // Debug mode off
      viewedItems.length > 0 ? [[...viewedItems]] : [[]] // Pass the array of viewed items
    );
    
    // Store the response in cache
    localStorage.setItem(PERSONALIZED_RECS_CACHE_KEY, JSON.stringify(response));
    
    // Update the cached viewed items
    localStorage.setItem(LAST_VIEWED_ITEMS_CACHE_KEY, JSON.stringify(viewedItems));
    
    // Dispatch event to notify components
    dispatchPersonalizedRecsUpdateEvent();
    
    return true;
  } catch (error) {
    console.error('Failed to update personalized recommendations', error);
    return false;
  }
}

/**
 * Check if the current user has personalized recommendations
 */
export function hasPersonalizedRecs(): boolean {
  return getCachedPersonalizedRecs() !== null;
}

/**
 * Clear personalized recommendations cache
 */
export function clearPersonalizedRecs(): void {
  if (typeof window === 'undefined') {
    return;
  }
  
  localStorage.removeItem(PERSONALIZED_RECS_CACHE_KEY);
  localStorage.removeItem(LAST_VIEWED_ITEMS_CACHE_KEY);
  
  // Dispatch event to notify components
  dispatchPersonalizedRecsUpdateEvent();
} 