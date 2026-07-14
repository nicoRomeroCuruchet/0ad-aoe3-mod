SHELL := /bin/bash

# Preserve runtime values literally before exporting them to recipes. In particular,
# this prevents GNU Make from evaluating functions embedded in command-line values.
override export STEPS := $(value STEPS)
override export EPISODES := $(value EPISODES)
override export ARGS := $(value ARGS)
override export MODEL := $(value MODEL)
override export TRUST_MODEL := $(value TRUST_MODEL)
override export NO_AGENT_VIEW := $(value NO_AGENT_VIEW)

.DEFAULT_GOAL := help

.PHONY: help setup test lint verify server oracle random train eval

help:
	@printf '%s\n' \
		'Common commands:' \
		'  make setup                         Install the locked Python environment' \
		'  make test                          Run tests with coverage' \
		'  make lint                          Run Ruff checks' \
		'  make verify                        Run all offline quality gates' \
		'  make server                        Start 0 A.D. with the RL interface' \
		'  make oracle [EPISODES=…] [ARGS=…]  Evaluate the oracle baseline' \
		'  make random [EPISODES=…] [ARGS=…]  Evaluate the random baseline' \
		'  make train [STEPS=…] [ARGS=…] [NO_AGENT_VIEW=1]  Train SAC' \
		'  make eval MODEL=… TRUST_MODEL=1    Evaluate a trusted SAC checkpoint'

setup:
	uv sync --locked

test:
	uv run --locked pytest --cov

lint:
	uv run --locked ruff check rl

verify: test lint
	uv run --locked python -m compileall -q rl
	uv pip check --python .venv/bin/python
	uv lock --check

server:
	./run_game.sh --rl-interface=127.0.0.1:6000

oracle:
	@episodes="$${EPISODES:-}"; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	uv run --locked python -m rl.eval --experiment rl/configs/m0_oracle.toml \
		"$${episode_args[@]}" "$${extra_args[@]}"

random:
	@episodes="$${EPISODES:-}"; \
	if [[ -n "$$episodes" && ! "$$episodes" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'EPISODES must be a positive integer.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	episode_args=(); \
	if [[ -n "$$episodes" ]]; then episode_args=(--episodes "$$episodes"); fi; \
	uv run --locked python -m rl.eval --experiment rl/configs/m0_random.toml \
		"$${episode_args[@]}" "$${extra_args[@]}"

train:
	@steps="$${STEPS:-}"; \
	no_agent_view="$${NO_AGENT_VIEW:-}"; \
	if [[ -n "$$steps" && ! "$$steps" =~ ^[1-9][0-9]*$$ ]]; then \
		printf '%s\n' 'STEPS must be a positive integer.' >&2; exit 2; \
	fi; \
	if [[ -n "$$no_agent_view" && "$$no_agent_view" != "1" ]]; then \
		printf '%s\n' 'NO_AGENT_VIEW must be 1 when set.' >&2; exit 2; \
	fi; \
	read -r -a extra_args <<< "$${ARGS:-}"; \
	if [[ "$$no_agent_view" == "1" ]]; then \
		for arg in "$${extra_args[@]}"; do \
			if [[ "$$arg" == "--agent-view" || "$$arg" == "--delay" || "$$arg" == --delay=* ]]; then \
				printf '%s\n' 'ARGS cannot include --agent-view or --delay when NO_AGENT_VIEW=1.' >&2; \
				exit 2; \
			fi; \
		done; \
	fi; \
	view_args=(--agent-view); \
	if [[ "$$no_agent_view" == "1" ]]; then view_args=(); fi; \
	step_args=(); \
	if [[ -n "$$steps" ]]; then step_args=(--timesteps "$$steps"); fi; \
	uv run --locked python -m rl.train --experiment rl/configs/m0_sb3_sac.toml \
		"$${view_args[@]}" "$${step_args[@]}" "$${extra_args[@]}"

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
	uv run --locked python -m rl.eval --experiment rl/configs/m0_sb3_sac.toml \
		--model "$$model" --trust-model "$${episode_args[@]}" "$${extra_args[@]}"
