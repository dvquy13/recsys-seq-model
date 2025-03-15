// src/app/book-details/page.tsx
"use client";

import React, { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import BookDetails from '@/components/BookDetails';
import Header from '@/components/Header';
import type { BookRecommendation } from '@/types/api';

function validateBookData(data: any): BookRecommendation {
  // Ensure all required fields are present and have correct types
  const validated: BookRecommendation = {
    image_url: String(data.image_url || ''),
    title: String(data.title || 'Untitled'),
    subtitle: String(data.subtitle || ''),
    average_rating: Number(data.average_rating) || 0,
    rating_number: Number(data.rating_number) || 0,
    price: String(data.price || '0'),
    score: Number(data.score) || 0,
    main_category: String(data.main_category || ''),
    parent_asin: String(data.parent_asin || ''),
  };

  // Log validation results for debugging
  console.log('Raw data:', data);
  console.log('Validated data:', validated);

  return validated;
}

function BookDetailsContent() {
  const searchParams = useSearchParams();
  const encodedData = searchParams.get('data');
  
  if (!encodedData) {
    console.error('No data parameter provided in URL');
    return (
      <div>
        <Header title="Error" />
        <div className="container mx-auto p-4">
          <p>No book data provided</p>
        </div>
      </div>
    );
  }
  
  try {
    console.log('Encoded data:', encodedData);
    const decodedData = decodeURIComponent(encodedData);
    console.log('Decoded data:', decodedData);
    
    const parsedData = JSON.parse(decodedData);
    console.log('Parsed data:', parsedData);
    
    const validatedData = validateBookData(parsedData);
    
    return (
      <div>
        <Header title="Book Details" />
        <BookDetails data={validatedData} />
      </div>
    );
  } catch (error) {
    console.error('Error parsing book data:', error);
    return (
      <div>
        <Header title="Error" />
        <div className="container mx-auto p-4">
          <p>Error parsing book data</p>
          <p className="text-sm text-gray-600 mt-2">Please try going back and selecting the book again.</p>
        </div>
      </div>
    );
  }
}

export default function BookDetailsPage() {
  return (
    <Suspense fallback={
      <div>
        <Header title="Loading..." />
        <div className="container mx-auto p-4">
          <p>Loading book details...</p>
        </div>
      </div>
    }>
      <BookDetailsContent />
    </Suspense>
  );
}