#!/bin/bash -e
docker run --rm --env REDIS_URL="$3" --env REDIS_PASSWD="$4" -v "$(pwd)":/aikit -v "$(pwd)"/.hypothesis:/.hypothesis khulnasoft/aikit:latest python3 -m pytest --backend "$1" aikit_tests/test_aikit/test_functional/test_core/"$2".py --tb=short
