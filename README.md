# Implement an MVP RecSys

# Prerequisite
- uv >= 0.6.3
- Docker

# Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Set up env var $ROOT_DIR: `export ROOT_DIR=$(pwd) && sed "s|^ROOT_DIR=.*|ROOT_DIR=$ROOT_DIR|" .env > .tmp && mv .tmp .env`
- Run `export $(grep -v '^#' .env | xargs)` to load the variables
- Run `uv sync --all-groups` to install the dependencies
- Run `chmod +x mlflow/wait-for-it.sh` to wait for MLflow to start before creating objects

# Train model
Run `make ml-platform-up` to start ML supporting services like MLFlow and Qdrant.

Run notebooks in this sequence denoted by the notebook name prefix. For example: 000 -> 001 -> 002...

Run notebook 020 to store the model outputs to Qdrant Vector Store.

Run notebook 021 to send supporting data to Redis to prepare for online serving.

# Run API
```shell
cd $ROOT_DIR
make requirements-txt
make api-up
echo "Visit http://localhost:8000/docs to interact with the APIs"
```

To test /score/seq_retriever endpoint, try this request body (feel free to change the actual item IDs):
```json
{
  "user_ids_raw": [""],
  "item_seq_raw": [
    ["B00DPM7TIG"]
  ],
  "candidate_items_raw": ["B00DPM7TIG"]
}
```

The main API endpoint is /recs/retrieve, try this request body:
```json
{
  "user_ids_raw": ["AE224PFXAEAT66IXX43GRJSWHXCA"],
  "item_seq_raw": [
    ["0439064864", "043935806X"]
  ],
  "candidate_items_raw": []
}
```

# Run Tests
The project includes comprehensive tests for the API. To run the tests:

```shell
# Run tests using the provided script
make test

# Or run manually
uv pip install --group testing
export PYTHONPATH=$(pwd)
pytest
```

For more details about the tests, see [tests/README.md](tests/README.md).
