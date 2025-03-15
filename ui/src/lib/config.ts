export const config = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  ui: {
    bookCoverImplementation: process.env.NEXT_PUBLIC_BOOK_COVER_IMPLEMENTATION || 'textBased',
    defaultRecommendationCount: 10,
  },
} as const;

// Type-safe config accessor
export function getConfig<
  K1 extends keyof typeof config,
  K2 extends keyof (typeof config)[K1]
>(key1: K1, key2: K2): (typeof config)[K1][K2] {
  return config[key1][key2];
} 