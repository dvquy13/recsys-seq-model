.PHONY:
.ONESHELL:

include .env
export

ml-platform-up:
	docker compose -f compose.yml up -d mlflow_server kv_store qdrant

ml-platform-logs:
# For make command that follows logs, if not add prefix '-' then when interrupet the command, it will complain with Error 130
	- docker compose -f compose.yml logs -f

lab:
	uv run jupyter lab --port 8888 --host 0.0.0.0

api-up:
	docker compose -f compose.api.yml up -d

api-logs:
	docker compose -f compose.api.yml logs -f

api-down:
	docker compose -f compose.api.yml down

# Create the requirements.txt file and update the torch to CPU version to reduce the image size
requirements-txt:
	uv export --group serving --group ml --no-hashes --format requirements-txt > requirements.txt
	# Commend out torch in requirements.txt to pre-install the CPU version in Docker
	sed '/^torch/ s/^/# /' requirements.txt > .tmp && mv .tmp requirements.txt
	sed '/^nvidia/ s/^/# /' requirements.txt > .tmp && mv .tmp requirements.txt

down:
	docker compose -f compose.yml down
	docker compose -f compose.api.yml down
