// components/BookDetails.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { BookCover, isPlaceholderImage } from './BookCover';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Card, CardContent } from "@/components/ui/card";
import type { BookRecommendation } from '@/types/api';

interface BookDetailsProps {
  data: BookRecommendation;
}

const BookDetails: React.FC<BookDetailsProps> = ({ data }) => {
  const router = useRouter();

  const handleBack = () => {
    router.back();
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <Button
        onClick={handleBack}
        variant="outline"
        className="mb-8"
      >
        ← Back to Recommendations
      </Button>

      <Card className="overflow-hidden">
        <CardContent className="p-0">
          <div className="flex flex-col md:flex-row">
            <div className="md:w-[300px] flex-shrink-0">
              {isPlaceholderImage(data.image_url) ? (
                <div className="w-full aspect-[3/4]">
                  <BookCover
                    title={data.title}
                    width={300}
                    height={400}
                    imageUrl={data.image_url}
                  />
                </div>
              ) : (
                <div className="relative w-full aspect-[3/4]">
                  <Image
                    src={data.image_url}
                    alt={data.title}
                    fill={true}
                    className="object-cover"
                  />
                </div>
              )}
            </div>

            <div className="flex-grow p-6">
              <div className="space-y-6">
                <div>
                  <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight">
                    {data.title}
                  </h1>
                  {data.subtitle && (
                    <p className="text-xl text-muted-foreground mt-2">
                      {data.subtitle}
                    </p>
                  )}
                </div>

                <Separator />

                <div className="grid gap-4">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="px-4 py-1">
                      Rating: {data.average_rating} ★
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      ({data.rating_number} reviews)
                    </span>
                  </div>

                  {data.price && data.price !== "None" && (
                    <div>
                      <Badge variant="outline" className="px-4 py-1">
                        ${data.price}
                      </Badge>
                    </div>
                  )}

                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Category:</span>
                      <Badge variant="secondary">
                        {data.main_category}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">ASIN:</span>
                      <code className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm">
                        {data.parent_asin}
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default BookDetails;