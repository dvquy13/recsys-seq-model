import '@testing-library/jest-dom'

// Mock Next.js components
jest.mock('next/image', () => ({
  __esModule: true,
  default: (props) => <img {...props} />,
})); 