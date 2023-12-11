from pytest import mark
from pathlib import Path


skip_ids = []
skips_path = Path(__file__).parent / "skips.txt"
if skips_path.exists():
    with open(skips_path) as f:
        for line in f:
            if line.startswith("aikit_tests"):
                id_ = line.strip("\n")
                skip_ids.append(id_)


def pytest_collection_modifyitems(items):
    skip_aikit = mark.skip(reason="aikit skip - see aikit_tests/skips.txt for details")
    for item in items:
        # skip if specified in skips.txt
        for id_ in skip_ids:
            if item.nodeid.startswith(id_):
                item.add_marker(skip_aikit)
                break
