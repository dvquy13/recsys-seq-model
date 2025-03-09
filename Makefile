.PHONY:
.ONESHELL:

include .env
export

model ?= SequenceRetriever
run_name ?= $(shell date +%Y%m%d%H%M%S)

train:
	mkdir -p notebooks/papermill-output
	cd notebooks && uv run papermill \
		011-sequence-modeling.ipynb \
		papermill-output/$(shell date +%Y%m%d%H%M%S)_011-sequence-modeling.ipynb \
		-p run_name $(run_name) \
		-p model_classname $(model) \
		-p max_epochs 100

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

clear-notebook-outputs:
	cd notebooks
	uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

down:
	docker compose -f compose.yml down
	docker compose -f compose.api.yml down
