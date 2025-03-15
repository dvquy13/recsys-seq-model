// src/app/book-details/page.tsx
"use client";

import React, { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import BookDetails from '@/components/BookDetails';
import Header from '@/components/Header';

function BookDetailsContent() {
  const searchParams = useSearchParams();
  const encodedData = searchParams.get('data');
  
  if (!encodedData) {
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
    const bookData = JSON.parse(decodeURIComponent(encodedData));
    
    return (
      <div>
        <Header title="Book Details" />
        <BookDetails data={bookData} />
      </div>
    );
  } catch {
    return (
      <div>
        <Header title="Error" />
        <div className="container mx-auto p-4">
          <p>Error parsing book data</p>
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