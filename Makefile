.DEFAULT_GOAL := help
VENV ?= $(HOME)/Documents/models/z-image-turbo/.venv
PYTHON := $(VENV)/bin/python
BOOTSTRAP_PYTHON ?= /opt/homebrew/bin/python3.14
PORT ?= 8000
NO_LORAS ?= 0
WITH_LORAS ?= 0
SERVER_FLAGS := $(if $(filter 1,$(NO_LORAS)),--no-loras,)
LORA_ID ?= realism

.PHONY: help setup setup-env download serve serve-no-loras start health smoke test install-lora download-loras
help:
	@echo 'make serve-no-loras  Start the base model with LoRAs disabled'
	@echo 'make serve     Start the local API (Ctrl-C stops it)'
	@echo 'make health    Check readiness'
	@echo 'make smoke     Generate and validate a 512px test image'
	@echo 'make download  Download/resume the pinned 4-bit model to this SSD'
	@echo 'make setup     Install dependencies and download the pinned model (one time)'
	@echo 'make setup WITH_LORAS=1  Also download the optional pinned LoRA adapters'
	@echo 'Override the port: make serve PORT=8001'

download-loras:
	"$(PYTHON)" download_loras.py

install-lora:
	@test -n "$(FILE)" || (echo 'Usage: make install-lora FILE=/absolute/path/to/ZImage.safetensors'; exit 1)
	"$(PYTHON)" install_lora.py "$(FILE)" --id "$(LORA_ID)"

setup: setup-env
	$(MAKE) download
	$(if $(filter 1,$(WITH_LORAS)),$(MAKE) download-loras,@:)

setup-env:
	@test -x "$(PYTHON)" || $(BOOTSTRAP_PYTHON) -m venv "$(VENV)"
	"$(PYTHON)" -m pip install --no-cache-dir -r requirements.lock.txt

download:
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
