// components/BookDetails.tsx
import React from 'react';
import { useRouter } from 'next/navigation';
import { Recommendation } from './RecommendationCard';

const BookDetails = ({ book }: { book: Recommendation }) => {
  const router = useRouter();

  const containerStyle: React.CSSProperties = {
    padding: '2rem',
    maxWidth: '800px',
    margin: '0 auto',
  };

  const detailsContainerStyle: React.CSSProperties = {
    display: 'flex',
    gap: '2rem',
    marginBottom: '2rem',
  };

  const imageStyle: React.CSSProperties = {
    width: '250px',
    height: '350px',
    objectFit: 'contain',
    boxShadow: '2px 2px 12px rgba(0,0,0,0.1)',
    backgroundColor: '#f8f9fa',
    padding: '8px',
  };

  const buttonStyle: React.CSSProperties = {
    padding: '0.5rem 1rem',
    fontSize: '1rem',
    cursor: 'pointer',
    backgroundColor: '#f8f9fa',
    border: '1px solid #dee2e6',
    borderRadius: '4px',
    marginTop: '1rem',
  };

  return (
    <div style={containerStyle}>
      <button 
        style={buttonStyle}
        onClick={() => router.back()}
      >
        ← Back to Recommendations
      </button>
      
      <div style={detailsContainerStyle}>
        <div>
          <img 
            src={book.image_url} 
            alt={book.title} 
            style={imageStyle}
          />
        </div>
        
        <div>
          <h1>{book.title}</h1>
          <h3>{book.subtitle}</h3>
          
          {book.author && <p><strong>Author:</strong> {book.author}</p>}
          
          <p>
            <strong>Rating:</strong> {book.average_rating} 
            <span style={{ color: '#666' }}> ({book.rating_number} reviews)</span>
          </p>
          
          <p><strong>Price:</strong> {book.price}</p>
          
          {book.isbn && <p><strong>ISBN:</strong> {book.isbn}</p>}
          
          {book.description && (
            <div>
              <h3>Description</h3>
              <p>{book.description}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BookDetails;