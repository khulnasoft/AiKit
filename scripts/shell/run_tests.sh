#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/aikit khulnasoft/aikit:latest python3 -m pytest aikit_tests/
