// src/components/Header.tsx
import React from 'react';

interface HeaderProps {
  title: string;
}

export default function Header({ title }: HeaderProps) {
  return (
    <header className="border-b">
      <div className="container px-4 py-6">
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
          {title}
        </h1>
      </div>
    </header>
  );
}