This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Recommendation Components

### RecommendationsGrid

The `RecommendationsGrid` component is a reusable component for displaying a grid of recommendations. It can be used to display different types of recommendation lists.

#### Usage

```tsx
import { RecommendationsGrid } from '@/components/RecommendationsGrid';

// In your component:
<RecommendationsGrid
  title="Recommendations"
  recommendations={recommendations}
  emptyMessage="No recommendations available"
/>
```

#### Props

- `title`: The title of the recommendations section
- `recommendations`: An array of recommendation objects
- `emptyMessage` (optional): Message to display when there are no recommendations
- `className` (optional): CSS class for the container
- `gridClassName` (optional): CSS class for the grid
- `titleClassName` (optional): CSS class for the title

### RecommendationCard

The `RecommendationCard` component displays a single recommendation. It has been updated to be more flexible for different use cases.

#### Props

- `recommendation`: A recommendation object
- `showScore` (optional): Whether to show the recommendation score (default: true)
- `linkPrefix` (optional): Prefix for the link URL (default: '/books')
- `className` (optional): CSS class for the card
