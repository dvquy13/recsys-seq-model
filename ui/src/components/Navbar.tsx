import Link from 'next/link'

export function Navbar() {
  return (
    <nav className="container mx-auto py-4">
      <div className="px-6">
        <Link href="/" className="text-lg font-semibold hover:text-primary">
          Home
        </Link>
      </div>
    </nav>
  )
} 