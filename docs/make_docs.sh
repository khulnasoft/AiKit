#!/bin/bash -e
docker run --rm -v "$(pwd)"/..:/project khulnasoft/doc-builder:latest
