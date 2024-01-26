#!/bin/bash

# shellcheck disable=SC2046
docker run --rm -v "$(pwd)":/aikit khulnasoft/aikit:latest python3 aikit/test_dependencies.py -fp aikit/requirements.txt,aikit/optional.txt
