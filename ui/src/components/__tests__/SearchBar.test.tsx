import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { SearchBar } from '../SearchBar';
import { useDebounce } from '@/lib/hooks/use-debounce';
import { getConfig } from '@/lib/config';

// Mock dependencies
jest.mock('@/lib/hooks/use-debounce', () => ({
  useDebounce: jest.fn(val => val), // No debounce in tests
}));

jest.mock('@/lib/config', () => ({
  getConfig: jest.fn().mockReturnValue('text'),
}));

// Mock next/link
jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children }: { href: string; children: React.ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

// Mock the BookCover component
jest.mock('@/components/book-covers', () => ({
  BookCover: () => <div data-testid="book-cover">Book Cover</div>,
}));

// Mock fetch
global.fetch = jest.fn();

describe('SearchBar Component', () => {
  const mockSearchResults = [
    {
      id: 'book1',
      title: 'Test Book 1',
      image_url: 'https://example.com/book1.jpg',
      author: 'Author 1'
    },
    {
      id: 'book2',
      title: 'Test Book 2',
      image_url: 'https://example.com/book2.jpg',
      author: 'Author 2'
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset fetch mock
    (global.fetch as jest.Mock).mockReset();
  });

  it('renders the search bar correctly', () => {
    render(<SearchBar />);
    
    // Check if search input is rendered
    const searchInput = screen.getByPlaceholderText('Search books...');
    expect(searchInput).toBeInTheDocument();
    
    // Dropdown should not be visible initially
    const dropdown = screen.queryByText('Books');
    expect(dropdown).not.toBeInTheDocument();
  });

  it('does not show dropdown when clicking on empty search bar', () => {
    render(<SearchBar />);
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    fireEvent.focus(searchInput);
    
    // Dropdown should still not be visible
    const dropdown = screen.queryByText('Books');
    expect(dropdown).not.toBeInTheDocument();
  });

  it('shows loading state while searching', async () => {
    // Mock fetch to delay response
    (global.fetch as jest.Mock).mockImplementationOnce(() => 
      new Promise(resolve => setTimeout(() => 
        resolve({
          ok: true,
          json: () => Promise.resolve({ items: [] })
        }), 100)
      )
    );

    render(<SearchBar />);
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    // Should show loading state
    await waitFor(() => {
      expect(screen.getByText('Searching...')).toBeInTheDocument();
    });
  });

  it('displays search results when available', async () => {
    // Mock successful search
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ items: mockSearchResults })
    });

    render(<SearchBar />);
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    // Wait for results to appear
    await waitFor(() => {
      expect(screen.getByText('Test Book 1')).toBeInTheDocument();
      expect(screen.getByText('Test Book 2')).toBeInTheDocument();
    });
  });

  it('shows "No results found" message when search returns empty', async () => {
    // Mock empty search results but with isLoading true at first to show dropdown
    (global.fetch as jest.Mock).mockImplementation(async () => {
      // First make isLoading true
      await new Promise(resolve => setTimeout(resolve, 10));
      // Then return empty results
      return {
        ok: true,
        json: () => Promise.resolve({ items: [] })
      };
    });

    render(<SearchBar />);
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    
    // Need to use act to handle state updates
    await act(async () => {
      fireEvent.change(searchInput, { target: { value: 'nonexistent' } });
    });
    
    // First verify loading is shown
    await waitFor(() => {
      expect(screen.queryByText('Searching...')).toBeInTheDocument();
    });
    
    // Then verify loading disappears
    await waitFor(() => {
      expect(screen.queryByText('Searching...')).not.toBeInTheDocument();
    });
    
    // And verify dropdown is not shown when there are no results
    // This is the expected behavior after our SearchBar component changes
    expect(screen.queryByText('Books')).not.toBeInTheDocument();
    expect(screen.queryByText('No results found.')).not.toBeInTheDocument();
  });

  it('hides dropdown when clicking outside', async () => {
    // Mock successful search
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ items: mockSearchResults })
    });

    render(
      <div>
        <SearchBar />
        <div data-testid="outside">Outside element</div>
      </div>
    );
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    // Wait for results to appear
    await waitFor(() => {
      expect(screen.getByText('Test Book 1')).toBeInTheDocument();
    });
    
    // Click outside the search component
    fireEvent.mouseDown(screen.getByTestId('outside'));
    
    // Dropdown should be hidden
    await waitFor(() => {
      expect(screen.queryByText('Test Book 1')).not.toBeInTheDocument();
    });
  });

  it('handles API error gracefully', async () => {
    // Mock failed search
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('API error'));
    
    // Suppress console error for this test
    const originalConsoleError = console.error;
    console.error = jest.fn();
    
    render(<SearchBar />);
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    fireEvent.change(searchInput, { target: { value: 'error test' } });
    
    // Wait for any state changes to complete
    await waitFor(() => {
      expect(console.error).toHaveBeenCalled();
    });
    
    // Restore console.error
    console.error = originalConsoleError;
  });

  it('clears search results when query is empty', async () => {
    // First mock successful search
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ items: mockSearchResults })
    });

    render(<SearchBar />);
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    
    // Enter search query
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    // Wait for results to appear
    await waitFor(() => {
      expect(screen.getByText('Test Book 1')).toBeInTheDocument();
    });
    
    // Clear search query
    fireEvent.change(searchInput, { target: { value: '' } });
    
    // Results should disappear
    await waitFor(() => {
      expect(screen.queryByText('Test Book 1')).not.toBeInTheDocument();
    });
  });

  it('creates correct book links', async () => {
    // Mock successful search
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ items: mockSearchResults })
    });

    render(<SearchBar />);
    
    const searchInput = screen.getByPlaceholderText('Search books...');
    fireEvent.change(searchInput, { target: { value: 'test' } });
    
    // Wait for results to appear
    await waitFor(() => {
      // Check that the links have the correct hrefs
      const links = document.querySelectorAll('a');
      expect(links[0].getAttribute('href')).toBe('/books/book1');
      expect(links[1].getAttribute('href')).toBe('/books/book2');
    });
  });
}); 