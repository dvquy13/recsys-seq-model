'use client';

import React from 'react';
import Image from 'next/image';
import { Card } from "@/components/ui/card";
import { createAvatar } from '@dicebear/core';
import { shapes } from '@dicebear/collection';
import { BookCoverProps } from './types';
import { isPlaceholderImage } from './TextBasedBookCover';

export const DiceBearBookCover: React.FC<BookCoverProps> = ({
  title,
  width = 200,
  height = 300,
  imageUrl,
}) => {
  const generateBookCover = (title: string): string => {
    const avatar = createAvatar(shapes, {
      seed: title,
      backgroundColor: ['b6e3f4','c0aede','d1d4f9','ffd5dc','ffdfbf'],
      backgroundType: ['gradientLinear'],
      backgroundRotation: [0, 360],
      shape1: ['ellipse', 'rectangle', 'polygonFilled'],
      shape2: ['ellipseFilled', 'rectangleFilled'],
      shape3: ['polygon', 'line'],
      shape1Color: ['ff0000', '00ff00', '0000ff'],
      shape2Color: ['ff0000', '00ff00', '0000ff'],
      shape3Color: ['ff0000', '00ff00', '0000ff'],
      scale: 90,
      size: Math.max(width, height),
      radius: 10
    });
  
    return avatar.toDataUri();
  };

  const hasValidImage = imageUrl && !isPlaceholderImage(imageUrl);
  const coverImage = hasValidImage ? imageUrl : generateBookCover(title);

  return (
    <Card
      className="relative overflow-hidden transition-shadow hover:shadow-lg"
      style={{
        width,
        height,
      }}
      data-testid="book-cover"
    >
      <Image
        src={coverImage}
        alt={title}
        className="object-contain w-full h-full"
        width={width}
        height={height}
      />
    </Card>
  );
}; 