"use client"

import Link from 'next/link'
import { Container, Wrapper } from '@/components/ui/container'
import { SearchBar } from '@/components/SearchBar'

export function Navbar() {
  return (
    <nav className="py-4">
      <Container>
        <Wrapper className="flex items-center justify-between">
          <Link href="/" className="text-lg font-semibold hover:text-primary">
            Home
          </Link>
          <div className="flex-1 flex justify-center mx-4">
            <SearchBar />
          </div>
          <div className="w-[100px]">
            {/* Placeholder to balance the navbar */}
          </div>
        </Wrapper>
      </Container>
    </nav>
  )
} 