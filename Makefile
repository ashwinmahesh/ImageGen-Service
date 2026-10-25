.DEFAULT_GOAL := help
VENV ?= $(CURDIR)/.venv
PYTHON := $(VENV)/bin/python
BOOTSTRAP_PYTHON ?= python3.14
PORT ?= 8000
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
SERVER_FLAGS := $(if $(filter 1,$(NO_LORAS)),--no-loras,)
LORA_ID ?= realism

.PHONY: storage help setup setup-selected setup-env download serve serve-no-loras start health smoke test install-lora download-loras download-realism
help:
	@echo 'make serve-no-loras  Start the base model with LoRAs disabled'
	@echo 'make serve     Choose a model and start the API (Ctrl-C stops it)'
	@echo 'make health    Check readiness'
	@echo 'make smoke     Generate and validate a 512px test image'
	@echo 'make download  Download/resume the selected model'
	@echo 'make setup     Choose a model, install dependencies and download it (one time)'
	@echo 'make setup WITH_REALISM=1  Also download the optional Realism adapter'
	@echo 'Choose explicitly: make serve MODEL=chroma1-flash PORT=8001'

download-realism:
	"$(PYTHON)" download_loras.py --id realism

download-loras:
	"$(PYTHON)" download_loras.py

install-lora:
	@test -n "$(FILE)" || (echo 'Usage: make install-lora FILE=/absolute/path/to/ZImage.safetensors'; exit 1)
	"$(PYTHON)" install_lora.py "$(FILE)" --id "$(LORA_ID)"

setup:
	"$(BOOTSTRAP_PYTHON)" setup.py

setup-selected: setup-env
	$(MAKE) download
	$(if $(filter 1,$(WITH_REALISM)),$(MAKE) download-realism,@:)

storage:
	"$(BOOTSTRAP_PYTHON)" storage_paths.py

setup-env: storage
	@test -x "$(PYTHON)" || $(BOOTSTRAP_PYTHON) -m venv "$(VENV)"
	"$(PYTHON)" -m pip install --no-cache-dir -r requirements.lock.txt

download: storage
	"$(PYTHON)" download.py

serve:
	"$(PYTHON)" launch.py --port $(PORT) $(SERVER_FLAGS)

serve-no-loras:
	$(MAKE) serve NO_LORAS=1

start: serve

health:
	@curl --fail --silent --show-error http://127.0.0.1:$(PORT)/health
	@echo

smoke:
	"$(PYTHON)" smoke_test.py --port $(PORT)

test:
	"$(PYTHON)" -m unittest discover -v
