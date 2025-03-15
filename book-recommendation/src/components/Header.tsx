// src/components/Header.tsx
import React from 'react';

interface HeaderProps {
  title: string;
}

export default function Header({ title }: HeaderProps) {
  return (
    <header className="py-4">
      <h1 className="text-4xl font-bold mb-4">{title}</h1>
    </header>
  );
}