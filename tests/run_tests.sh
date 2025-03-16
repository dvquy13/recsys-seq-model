#!/bin/bash

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

# Create a symlink to the mock idm.json file
ln -sf $(pwd)/tests/mock_idm.json $(pwd)/idm.json

# Run tests with coverage
uv run pytest --cov=api --cov-report=term-missing

# Remove the symlink
rm -f $(pwd)/idm.json 