import sys
from pymongo import MongoClient

action_url = "https://github.com/khulnasoft/aikit/actions/runs/"

test_configs = {
    "test-array-api": ["array_api", 0],
    "test-core-aikit": ["aikit_core", 1],
    "test-nn-aikit": ["aikit_nn", 2],
    "test-stateful-aikit": ["aikit_stateful", 3],
    "test-frontend-tensorflow-push": ["tf_frontend", 4],
    "test-frontend-numpy-push": ["numpy_frontend", 5],
    "test-frontend-jax-push": ["jax_frontend", 6],
    "test-frontend-torch-push": ["torch_frontend", 7],
    "test-experimental-core-aikit": ["experimental_core", 8],
    "test-experimental-nn-aikit": ["experimental_nn", 9],
}
result_config = {
    "success": "https://img.shields.io/badge/-success-success",
    "failure": "https://img.shields.io/badge/-failure-red",
}


def make_clickable(url, name):
    return (
        f'<a href="{url}" rel="noopener noreferrer" '
        + f'target="_blank"><img src={name}></a>'
    )


def update_test_results():
    key, workflow, fw_submod, result, run_id = (
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        str(sys.argv[5]),
    )
    backend = fw_submod.split("-")[0]
    submodule = fw_submod.split("-")[1]
    cluster = MongoClient(
        f"mongodb+srv://deep-aikit:{key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["Aikit_tests_multi"]
    collection = db[test_configs[workflow][0]]
    res = make_clickable(action_url + run_id, result_config[result])
    collection.update_one(
        {"_id": test_configs[workflow][1]},
        {"$set": {f"{backend}.{submodule}": res}},
        upsert=True,
    )
    return


if __name__ == "__main__":
    update_test_results()
