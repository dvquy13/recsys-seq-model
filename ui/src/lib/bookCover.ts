import { createAvatar } from '@dicebear/core';
import { shapes } from '@dicebear/collection';

export function generateBookCover(title: string): string {
  // Use the title as seed to generate consistent avatars for the same title
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
    size: 200,
    radius: 10
  });

  return avatar.toDataUri();
}

export function shouldGenerateBookCover(imageUrl: string | undefined): boolean {
  if (!imageUrl) return true;
  return imageUrl.includes('placehold.co');
} 