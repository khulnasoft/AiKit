#!/bin/bash

docker build -t khulnasoft/aikit:latest --no-cache -f Dockerfile ..
docker build -t khulnasoft/aikit:latest-gpu --no-cache -f DockerfileGPU ..
