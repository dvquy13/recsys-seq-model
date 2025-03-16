# API Tests

This directory contains tests for the API.

## Structure

- `conftest.py`: Contains fixtures for testing the API
- `api/test_endpoints.py`: Tests for the API endpoints
- `api/test_services.py`: Tests for the service layer
- `api/test_models.py`: Tests for the Pydantic models

## Running Tests

To run the tests, use the following command:

```bash
# Install dependencies and run tests (recommended)
./tests/run_tests.sh

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=api --cov-report=term-missing

# Run specific tests
uv run pytest tests/api/test_endpoints.py
uv run pytest tests/api/test_services.py
uv run pytest tests/api/test_models.py
```

## Test Coverage

The tests cover:

1. **API Endpoints**:
   - `/recs/retrieve`: Retrieve recommendations
   - `/recs/popular`: Get popular recommendations
   - `/vendor/seq_retriever`: Call the sequence retriever
   - `/items/get_by_ids`: Get items by IDs

2. **Service Layer**:
   - `RecommendationService`: Service for handling recommendations

3. **Models**:
   - Pydantic models for request and response validation

## Mocking

The tests use mocking to isolate the components being tested:

- Redis client is mocked to return predefined data
- Qdrant client is mocked to return predefined search results
- ID mapper is mocked to convert item IDs to indices
- External HTTP calls are mocked to return predefined responses

## Dependencies

Test dependencies are managed through the `testing` dependency group in `pyproject.toml`. This ensures consistent dependency management with the rest of the project using `uv`. 