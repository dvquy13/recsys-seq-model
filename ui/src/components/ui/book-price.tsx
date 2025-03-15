import { Badge } from '@/components/ui/badge';

interface BookPriceProps {
  price: string | null;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function BookPrice({ 
  price, 
  size = 'md',
  className = '' 
}: BookPriceProps) {
  // If price is null or undefined, don't render anything
  if (price === null || price === undefined) {
    return null;
  }

  // Size configurations
  const config = {
    sm: 'text-xs py-0.5 px-1.5',
    md: 'text-sm py-0.5 px-2',
    lg: 'text-base py-1 px-2.5',
  };

  const sizeClass = config[size];
  
  // Check if the book is free
  const isFree = price === "0.0" || price === "0" || parseFloat(price) === 0;
  
  return (
    <Badge 
      variant="outline" 
      className={`${sizeClass} ${className}`}
    >
      {isFree ? 'Free' : `$${price}`}
    </Badge>
  );
} 