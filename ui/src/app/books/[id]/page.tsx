import { Metadata } from 'next';
import Image from 'next/image';
import { notFound } from 'next/navigation';
import { getBookDetails } from '@/lib/api';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { BookCover } from '@/components/book-covers';
import { getConfig } from '@/lib/config';
import { RecommendationsGrid } from '@/components/RecommendationsGrid';
import { BookRating } from '@/components/ui/book-rating';
import { BookPrice } from '@/components/ui/book-price';

interface BookPageProps {
  params: {
    id: string;
  };
}

export async function generateMetadata({ params }: BookPageProps): Promise<Metadata> {
  try {
    const book = await getBookDetails(params.id);
    return {
      title: book.title,
      description: book.subtitle || `Details for ${book.title}`,
    };
  } catch (error) {
    return {
      title: 'Book Not Found',
      description: 'The requested book could not be found',
    };
  }
}

export default async function BookPage({ params }: BookPageProps) {
  let book;
  try {
    book = await getBookDetails(params.id);
  } catch (error) {
    notFound();
    return undefined;
  }

  const bookCoverImplementation = getConfig('ui', 'bookCoverImplementation');

  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="max-w-4xl mx-auto">
        <CardHeader>
          <div className="flex flex-col md:flex-row gap-8">
            <div className="relative w-full md:w-[300px] h-[400px]">
              <BookCover
                title={book.title}
                imageUrl={book.image_url}
                width={300}
                height={400}
                parent_asin={book.parent_asin}
                implementation={bookCoverImplementation}
              />
            </div>
            <div className="flex-1 space-y-4">
              <h1 className="text-3xl font-bold">{book.title}</h1>
              {book.subtitle && (
                <p className="text-lg text-muted-foreground">{book.subtitle}</p>
              )}
              
              <div className="flex items-center">
                <BookRating 
                  rating={book.average_rating} 
                  ratingCount={book.rating_number}
                  size="lg"
                />
              </div>
              
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">{book.main_category}</Badge>
                <BookPrice price={book.price} size="md" />
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Additional book details can be added here */}
          
          {/* This is where future recommendation sections will be added */}
          {/* Example:
          <div className="mt-8">
            <RecommendationsGrid 
              title="Similar Books" 
              recommendations={similarBooks} 
            />
          </div>
          */}
        </CardContent>
      </Card>
    </div>
  );
} 