// src/app/book-details/page.tsx
"use client";

import { useSearchParams } from 'next/navigation';
import BookDetails from '../../components/BookDetails';
import Header from '../../components/Header';

export default function BookDetailsPage() {
  const searchParams = useSearchParams();
  const encodedData = searchParams.get('data');
  
  if (!encodedData) {
    return <div>No book details available</div>;
  }
  
  try {
    // First, try to safely decode the URI component
    const decodedData = decodeURIComponent(encodedData);
    
    // Handle potential control characters but preserve newlines
    const sanitizedData = decodedData.replace(/[\u0000-\u001F\u007F-\u009F]/g, function(match) {
      // Preserve newline characters in the JSON string
      if (match === '\n' || match === '\\n') return match;
      return '';
    });
    
    const bookData = JSON.parse(sanitizedData);
    
    return (
      <div>
        <Header title="Book Details" />
        <BookDetails book={bookData} />
      </div>
    );
  } catch (error) {
    console.error('Error parsing book data:', error);
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <h2>Error loading book details</h2>
        <p>There was a problem parsing the book data.</p>
        <button 
          onClick={() => window.history.back()}
          style={{ 
            padding: '0.5rem 1rem', 
            fontSize: '1rem', 
            cursor: 'pointer',
            marginTop: '1rem' 
          }}
        >
          ← Back to Recommendations
        </button>
      </div>
    );
  }
}