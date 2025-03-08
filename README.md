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
Run notebooks in this sequence denoted by the notebook name prefix. For example: 000 -> 001 -> 002...
