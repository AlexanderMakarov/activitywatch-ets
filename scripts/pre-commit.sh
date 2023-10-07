#!/usr/bin/env bash

echo "Running $0"

# Note that need to be executed with python in "venv" folder to have all modules installed.
set -xe
.venv/bin/python test.py
