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
import { BookViewTracker } from '@/components/BookViewTracker';

interface BookPageProps {
  params: {
    id: string;
  };
}

export async function generateMetadata({ params }: BookPageProps): Promise<Metadata> {
  try {
    const resolvedParams = await params;
    const bookId = decodeURIComponent(resolvedParams.id);
    const book = await getBookDetails(bookId);
    
    if (!book) {
      return {
        title: 'Book Not Found',
        description: 'The requested book could not be found',
      };
    }
    
    return {
      title: book.title || 'Book Details',
      description: book.subtitle || `Details for ${book.title || 'this book'}`,
    };
  } catch (error) {
    console.error('Error generating metadata:', error);
    return {
      title: 'Book Not Found',
      description: 'The requested book could not be found',
    };
  }
}

export default async function BookPage({ params }: BookPageProps) {
  try {
    const resolvedParams = await params;
    const bookId = decodeURIComponent(resolvedParams.id);
    console.log('Fetching book details for ID:', bookId);
    
    const book = await getBookDetails(bookId);
    
    if (!book) {
      console.error('Book not found:', bookId);
      notFound();
      return null;
    }
    
    console.log('Book details:', book);
    
    const bookCoverImplementation = getConfig('ui', 'bookCoverImplementation');

    return (
      <div className="container mx-auto py-8 px-4">
        <BookViewTracker book={book} />
        
        <Card className="max-w-4xl mx-auto">
          <CardHeader>
            <div className="flex flex-col md:flex-row gap-8">
              <div className="relative w-full md:w-[300px] h-[400px]">
                <BookCover
                  title={book.title || 'Book Cover'}
                  imageUrl={book.image_url}
                  width={300}
                  height={400}
                  parent_asin={book.parent_asin}
                  implementation={bookCoverImplementation}
                />
              </div>
              <div className="flex-1 space-y-4">
                <h1 className="text-3xl font-bold">{book.title || 'Untitled Book'}</h1>
                {book.subtitle && (
                  <p className="text-lg text-muted-foreground">{book.subtitle}</p>
                )}
                
                <div className="flex items-center">
                  <BookRating 
                    rating={book.average_rating || 0} 
                    ratingCount={book.rating_number || 0}
                    size="lg"
                  />
                </div>
                
                <div className="flex flex-wrap gap-2">
                  {book.main_category && (
                    <Badge variant="secondary">{book.main_category}</Badge>
                  )}
                  {book.price !== undefined && (
                    <BookPrice price={book.price} size="md" />
                  )}
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
  } catch (error) {
    console.error('Error rendering book page:', error);
    notFound();
    return null;
  }
} 