from pymongo import MongoClient
from get_all_tests import BACKENDS


def main():
    # connect to the database
    cluster = MongoClient(
        "mongodb+srv://readonly-user:hvpwV5yVeZdgyTTm@cluster0.qdvf8q3.mongodb.net"
    )
    ci_dashboard_db = cluster["ci_dashboard"]
    aikit_tests_collection = ci_dashboard_db["aikit_tests"]
    frontend_tests_collection = ci_dashboard_db["frontend_tests"]
    demos_collection = ci_dashboard_db["demos"]

    # iterate over demos and collect aikit and frontend functions used
    demos = demos_collection.find()
    aikit_functions, frontend_functions = [], []
    for demo in demos:
        aikit_functions += demo.get("aikit_functions", [])
        frontend_functions += demo.get("frontend_functions", [])
    aikit_functions = list(set(aikit_functions))
    frontend_functions = list(set(frontend_functions))

    # find corresponding test paths for those functions
    aikit_test_paths = []
    frontend_test_paths = []
    for function in aikit_functions:
        result = aikit_tests_collection.find_one({"_id": function})
        if result:
            aikit_test_paths.append(result["test_path"])
    for function in frontend_functions:
        result = frontend_tests_collection.find_one({"_id": function})
        if result:
            frontend_test_paths.append(result["test_path"])

    # add those paths to the tests_to_run
    with open("tests_to_run", "w") as write_file:
        for test_path in aikit_test_paths + frontend_test_paths:
            test_path = test_path.strip()
            for backend in BACKENDS:
                write_file.write(f"{test_path},{backend}\n")


if __name__ == "__main__":
    main()
