SHELL := /bin/bash

UV ?= $(shell command -v uv 2>/dev/null || find "$$HOME/snap/code" -path '*/.local/bin/uv' -type f -perm -u+x 2>/dev/null | sort -V | tail -n 1)
UV_REQUIRED = command -v "$(UV)" >/dev/null 2>&1 || { printf '%s\n' 'uv is required to create and run the locked Python 3.11 RL environment.' 'Ubuntu 22.04 does not package Astral uv directly in the default apt repos.' 'Install with: sudo apt install pipx && pipx install uv' 'If the standalone installer used VS Code Snap, source the env file it printed.' 'Finally run: make setup' >&2; exit 127; }

# Preserve runtime values literally before exporting them to recipes. In particular,
# this prevents GNU Make from evaluating functions embedded in command-line values.
override export STEPS := $(value STEPS)
override export EPISODES := $(value EPISODES)
override export ARGS := $(value ARGS)
override export MODEL := $(value MODEL)
override export TRUST_MODEL := $(value TRUST_MODEL)

.DEFAULT_GOAL := help

.PHONY: help setup test lint verify engine-observer server server-view oracle random train ppo-train eval m1-oracle m1-random m1-train m1-ppo-train m1-eval

help:
	@printf '%s\n' \
		'Common commands:' \
		'  make setup                         Install the locked Python environment' \
		'  make test                          Run tests with coverage' \
		'  make lint                          Run Ruff checks' \
		'  make verify                        Run all offline quality gates' \
		'  make engine-observer                Build patched 0 A.D. rendered-view engine' \
		'  make server                        Start headless 0 A.D. for training' \
		'  make server-view                   Start visual 0 A.D. for debugging' \
		'  make oracle [EPISODES=…] [ARGS=…]  Evaluate the oracle baseline' \
		'  make random [EPISODES=…] [ARGS=…]  Evaluate the random baseline' \
		'  make train [STEPS=…] [MODEL=… TRUST_MODEL=1] [ARGS=…]  Train/resume SAC' \
		'  make ppo-train [STEPS=…] [MODEL=… TRUST_MODEL=1] [ARGS=…]  Train/resume PPO' \
		'  make eval MODEL=… TRUST_MODEL=1    Evaluate a trusted SAC checkpoint' \
		'  make m1-oracle [EPISODES=…]        Evaluate M1 stock-reward oracle' \
		'  make m1-train [STEPS=…] [MODEL=… TRUST_MODEL=1] [ARGS="--log-interval 1"]  Train/resume SAC on M1 stock reward' \
		'  make m1-ppo-train [STEPS=…] [MODEL=… TRUST_MODEL=1]  Train/resume PPO on M1 stock reward' \
		'  make m1-eval MODEL=… TRUST_MODEL=1 Evaluate an M1 checkpoint'

setup:
	@$(UV_REQUIRED)
	"$(UV)" sync --locked

test:
	@$(UV_REQUIRED)
	"$(UV)" run --locked pytest --cov

lint:
	@$(UV_REQUIRED)
	"$(UV)" run --locked ruff check rl

verify: test lint
	@$(UV_REQUIRED)
	"$(UV)" run --locked python -m compileall -q rl
	"$(UV)" pip check --python .venv/bin/python
	"$(UV)" lock --check

engine-observer:
	./engine/build_observer.sh

server:
	./run_game.sh --require-rl-observer -autostart-nonvisual --rl-interface=127.0.0.1:6000

server-view:
	./run_game.sh --require-rl-observer --rl-interface=127.0.0.1:6000

oracle:
	@episodes="$${EPISODES:-}"; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.eval --experiment rl/configs/m0_oracle.toml \
		"$${episode_args[@]}" "$${extra_args[@]}"

random:
	@episodes="$${EPISODES:-}"; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.eval --experiment rl/configs/m0_random.toml \
		"$${episode_args[@]}" "$${extra_args[@]}"

train:
	@steps="$${STEPS:-}"; \
	model="$${MODEL:-}"; \
	trust_model="$${TRUST_MODEL:-}"; \
	if [[ -n "$$steps" && ! "$$steps" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'STEPS must be a positive integer.' >&2; exit 2; \
	fi; \
	if [[ -n "$$model" && "$$trust_model" != "1" ]]; then \
		printf '%s\n' 'Refusing to deserialize the checkpoint without explicit TRUST_MODEL=1.' >&2; \
		exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	step_args=(); \
	if [[ -n "$$steps" ]]; then step_args=(--timesteps "$$steps"); fi; \
	resume_args=(); \
	if [[ -n "$$model" ]]; then resume_args=(--resume-from "$$model" --trust-model); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.train --experiment rl/configs/m0_sb3_sac.toml \
		"$${step_args[@]}" "$${resume_args[@]}" "$${extra_args[@]}"

ppo-train:
	@steps="$${STEPS:-}"; \
	model="$${MODEL:-}"; \
	trust_model="$${TRUST_MODEL:-}"; \
	if [[ -n "$$steps" && ! "$$steps" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'STEPS must be a positive integer.' >&2; exit 2; \
	fi; \
	if [[ -n "$$model" && "$$trust_model" != "1" ]]; then \
		printf '%s\n' 'Refusing to deserialize the checkpoint without explicit TRUST_MODEL=1.' >&2; \
		exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	step_args=(); \
	if [[ -n "$$steps" ]]; then step_args=(--timesteps "$$steps"); fi; \
	resume_args=(); \
	if [[ -n "$$model" ]]; then resume_args=(--resume-from "$$model" --trust-model); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.train --experiment rl/configs/m0_sb3_ppo.toml \
		"$${step_args[@]}" "$${resume_args[@]}" "$${extra_args[@]}"

eval:
	@model="$${MODEL:-}"; \
	trust_model="$${TRUST_MODEL:-}"; \
	episodes="$${EPISODES:-}"; \
	if [[ -z "$$model" ]]; then \
		printf '%s\n' 'Usage: make eval MODEL=path/to/model TRUST_MODEL=1 [EPISODES=1] [ARGS="--mode both"]' >&2; \
		exit 2; \
	fi; \
	if [[ "$$trust_model" != "1" ]]; then \
		printf '%s\n' 'Refusing to deserialize the checkpoint without explicit TRUST_MODEL=1.' >&2; \
		exit 2; \
	fi; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.eval --experiment rl/configs/m0_sb3_sac.toml \
		--model "$$model" --trust-model "$${episode_args[@]}" "$${extra_args[@]}"

m1-oracle:
	@episodes="$${EPISODES:-}"; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.eval --experiment rl/configs/m1_oracle.toml \
		"$${episode_args[@]}" "$${extra_args[@]}"

m1-random:
	@episodes="$${EPISODES:-}"; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.eval --experiment rl/configs/m1_random.toml \
		"$${episode_args[@]}" "$${extra_args[@]}"

m1-train:
	@steps="$${STEPS:-}"; \
	model="$${MODEL:-}"; \
	trust_model="$${TRUST_MODEL:-}"; \
	if [[ -n "$$steps" && ! "$$steps" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'STEPS must be a positive integer.' >&2; exit 2; \
	fi; \
	if [[ -n "$$model" && "$$trust_model" != "1" ]]; then \
		printf '%s\n' 'Refusing to deserialize the checkpoint without explicit TRUST_MODEL=1.' >&2; \
		exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	step_args=(); \
	if [[ -n "$$steps" ]]; then step_args=(--timesteps "$$steps"); fi; \
	resume_args=(); \
	if [[ -n "$$model" ]]; then resume_args=(--resume-from "$$model" --trust-model); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.train --experiment rl/configs/m1_sb3_sac.toml \
		"$${step_args[@]}" "$${resume_args[@]}" "$${extra_args[@]}"

m1-ppo-train:
	@steps="$${STEPS:-}"; \
	model="$${MODEL:-}"; \
	trust_model="$${TRUST_MODEL:-}"; \
	if [[ -n "$$steps" && ! "$$steps" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'STEPS must be a positive integer.' >&2; exit 2; \
	fi; \
	if [[ -n "$$model" && "$$trust_model" != "1" ]]; then \
		printf '%s\n' 'Refusing to deserialize the checkpoint without explicit TRUST_MODEL=1.' >&2; \
		exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	step_args=(); \
	if [[ -n "$$steps" ]]; then step_args=(--timesteps "$$steps"); fi; \
	resume_args=(); \
	if [[ -n "$$model" ]]; then resume_args=(--resume-from "$$model" --trust-model); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.train --experiment rl/configs/m1_sb3_ppo.toml \
		"$${step_args[@]}" "$${resume_args[@]}" "$${extra_args[@]}"

m1-eval:
	@model="$${MODEL:-}"; \
	trust_model="$${TRUST_MODEL:-}"; \
	episodes="$${EPISODES:-}"; \
	if [[ -z "$$model" ]]; then \
		printf '%s\n' 'Usage: make m1-eval MODEL=path/to/model TRUST_MODEL=1 [EPISODES=1] [ARGS="--mode both"]' >&2; \
		exit 2; \
	fi; \
	if [[ "$$trust_model" != "1" ]]; then \
		printf '%s\n' 'Refusing to deserialize the checkpoint without explicit TRUST_MODEL=1.' >&2; \
		exit 2; \
	fi; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	$(UV_REQUIRED); \
	"$(UV)" run --locked python -m rl.eval --experiment rl/configs/m1_sb3_sac.toml \
		--model "$$model" --trust-model "$${episode_args[@]}" "$${extra_args[@]}"
