"use client"

import Link from 'next/link'
import { Container, Wrapper } from '@/components/ui/container'

export function Navbar() {
  return (
    <nav className="py-4">
      <Container>
        <Wrapper className="flex items-center justify-between">
          <Link href="/" className="text-lg font-semibold hover:text-primary">
            Home
          </Link>
        </Wrapper>
      </Container>
    </nav>
  )
} 