export interface BookCoverProps {
  title: string;
  author?: string;
  width?: number;
  height?: number;
  backgroundColor?: string;
  textColor?: string;
  imageUrl?: string;
  parent_asin?: string;
}

export interface BookCoverImplementation {
  name: string;
  component: React.ComponentType<BookCoverProps>;
  description: string;
} 