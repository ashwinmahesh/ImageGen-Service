.DEFAULT_GOAL := help
VENV ?= $(CURDIR)/.venv
export VENV
PYTHON := $(VENV)/bin/python
BOOTSTRAP_PYTHON ?=
export BOOTSTRAP_PYTHON
BOOTSTRAP := sh ./bootstrap_python.sh
PORT ?= 5732
QUEUE_PORT ?= 55302
MODEL ?=
ifneq ($(strip $(MODEL)),)
export IMAGE_MODEL := $(MODEL)
endif
DATA_DIR ?=
ifneq ($(strip $(DATA_DIR)),)
export DATA_DIR
endif
NO_LORAS ?= 0
WITH_REALISM ?= 0
NO_LLM ?= 0
ifneq ($(filter-out 0 1,$(strip $(NO_LLM))),)
$(error NO_LLM accepts only 0 or 1)
endif
SERVER_FLAGS := $(if $(filter 1,$(NO_LORAS)),--no-loras,) $(if $(filter 1,$(NO_LLM)),--no-llm,)
LORA_ID ?= realism

.PHONY: storage help setup setup-selected setup-env download serve serve-no-loras start health smoke test install-lora download-loras download-realism download-llm queue stop-queue
help:
	@echo 'Setup reuses compatible Python, or installs it inside the project; serving stays offline'
	@echo 'make serve-no-loras  Start the base model with LoRAs disabled'
	@echo 'make queue     Start/reuse the lightweight queue API on localhost:55302 (no model/GPU)'
	@echo 'make stop-queue  Stop only the queue API; retain all persisted jobs'
	@echo 'make serve     Start the generator/UI on localhost:5732; starts queue API if needed'
	@echo 'Ctrl-C stops the generator after active work finishes; queue API stays running'
	@echo 'Prompt LLM defaults ON; interactive startup asks [Y/n]'
	@echo 'make serve NO_LLM=1  Disable local prompt expansion explicitly'
	@echo 'make download-llm  Download/resume the shared 2B prompt LLM'
	@echo 'make health    Check readiness'
	@echo 'make smoke     Generate and validate a 512px test image'
	@echo 'make download  Download/resume the selected model'
	@echo 'make setup     Choose a model, install dependencies and download image + prompt models (safe to rerun)'
	@echo 'Setup: q cancels at the model menu; Ctrl-C cancels at any time and keeps existing downloads'
	@echo 'make setup WITH_REALISM=1  Also download the optional Realism adapter'
	@echo 'Choose explicitly: make serve MODEL=chroma1-flash PORT=5733'
	@echo 'Photo editing models: flux2-klein-4b, flux2-klein-9b, qwen-image-2.1-turbo'

download-realism:
	"$(PYTHON)" download_loras.py --id realism

download-loras:
	"$(PYTHON)" download_loras.py

install-lora:
	@test -n "$(FILE)" || (echo 'Usage: make install-lora FILE=/absolute/path/to/ZImage.safetensors'; exit 1)
	"$(PYTHON)" install_lora.py "$(FILE)" --id "$(LORA_ID)"

setup:
	@$(BOOTSTRAP) setup setup.py

setup-selected: setup-env
	$(MAKE) download
	$(MAKE) download-llm
	$(if $(filter 1,$(WITH_REALISM)),$(MAKE) download-realism,@:)

storage:
	"$(PYTHON)" storage_paths.py

setup-env:
	@$(BOOTSTRAP) setup storage_paths.py
	@$(BOOTSTRAP) setup runtime_setup.py

download-llm: storage
	"$(PYTHON)" download_enhancer.py

download: storage
	"$(PYTHON)" download.py

queue:
	"$(PYTHON)" queue_launch.py --port $(QUEUE_PORT)

stop-queue:
	"$(PYTHON)" queue_launch.py --port $(QUEUE_PORT) --stop

serve:
	"$(PYTHON)" launch.py --port $(PORT) --queue-port $(QUEUE_PORT) $(SERVER_FLAGS)

serve-no-loras:
	$(MAKE) serve NO_LORAS=1

start: serve

health:
	@curl --fail --silent --show-error http://127.0.0.1:$(PORT)/health
	@echo

smoke:
	"$(PYTHON)" -m tests.smoke_test --port $(PORT)

test:
	"$(PYTHON)" -m unittest discover -s tests -t . -v
