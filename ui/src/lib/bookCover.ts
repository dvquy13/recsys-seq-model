export function generateBookCover(title: string): string {
  // Generate a placeholder URL with the book title
  const encodedTitle = encodeURIComponent(title);
  return `https://placehold.co/400x600/e2e8f0/1e293b?text=${encodedTitle}`;
}

export function shouldGenerateBookCover(imageUrl: string | undefined): boolean {
  if (!imageUrl) return true;
  return imageUrl.includes('placehold.co');
} 