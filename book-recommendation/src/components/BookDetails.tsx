// components/BookDetails.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { BookCover, isPlaceholderImage } from './BookCover';

interface BookDetailsProps {
  data: {
    image_url: string;
    title: string;
    subtitle: string;
    average_rating: number;
    rating_number: number;
    price: string;
    description?: string;
    author?: string;
    isbn?: string;
  };
}

const BookDetails: React.FC<BookDetailsProps> = ({ data }) => {
  const router = useRouter();

  const handleBack = () => {
    router.back();
  };

  return (
    <div className="container mx-auto p-4">
      <button
        onClick={handleBack}
        className="mb-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Back to Recommendations
      </button>
      <div className="flex flex-col md:flex-row gap-8">
        <div className="flex-shrink-0">
          {isPlaceholderImage(data.image_url) ? (
            <div style={{ width: '300px', height: '400px' }}>
              <BookCover
                title={data.title}
                author={data.author}
                width={300}
                height={400}
              />
            </div>
          ) : (
            <div style={{ position: 'relative', width: '300px', height: '400px' }}>
              <Image
                src={data.image_url}
                alt={data.title}
                fill
                style={{ objectFit: 'cover' }}
              />
            </div>
          )}
        </div>
        <div className="flex-grow">
          <h1 className="text-3xl font-bold mb-2">{data.title}</h1>
          {data.author && (
            <h2 className="text-xl text-gray-600 mb-4">by {data.author}</h2>
          )}
          <p className="text-lg mb-4">{data.subtitle}</p>
          <div className="mb-4">
            <span className="font-bold">Rating: </span>
            {data.average_rating} ({data.rating_number} reviews)
          </div>
          <div className="mb-4">
            <span className="font-bold">Price: </span>
            {data.price}
          </div>
          {data.description && (
            <div>
              <h3 className="text-xl font-bold mb-2">Description</h3>
              <p className="text-gray-700">{data.description}</p>
            </div>
          )}
          {data.isbn && (
            <div className="mt-4">
              <span className="font-bold">ISBN: </span>
              {data.isbn}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BookDetails;