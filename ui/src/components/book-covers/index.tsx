'use client';

import React from 'react';
import { BookCoverProps, BookCoverImplementation } from './types';
import { TextBasedBookCover } from './TextBasedBookCover';
import { DiceBearBookCover } from './DiceBearBookCover';

export const implementations: BookCoverImplementation[] = [
  {
    name: 'textBased',
    component: TextBasedBookCover,
    description: 'A text-based book cover with patterns',
  },
  {
    name: 'dicebear',
    component: DiceBearBookCover,
    description: 'A book cover using DiceBear shapes',
  },
];

export interface BookCoverWrapperProps extends BookCoverProps {
  implementation?: string;
}

export const BookCover: React.FC<BookCoverWrapperProps> = ({
  implementation = 'textBased',
  ...props
}) => {
  const selectedImpl = implementations.find(impl => impl.name === implementation) || implementations[0];
  const Component = selectedImpl.component;
  
  return <Component {...props} />;
};

// Re-export types and implementations
export type { BookCoverProps } from './types';
export { isPlaceholderImage } from './TextBasedBookCover'; 