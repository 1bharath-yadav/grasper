import requests
import logging

logging.basicConfig(level=logging.INFO)

API_URL = "http://localhost:8000/api/analyze"
TEST_FILES = [
    "tests/sample-network/question.txt",
    "tests/sample-sales/question.txt",
    "tests/sample-weather/question.txt"
]


def run_all_tests():
    for test_file in TEST_FILES:
        with open(test_file, "rb") as f:
            files = {"questions.txt": f}
            logging.info(f"[TEST] Running test for: {test_file}")
            response = requests.post(API_URL, files=files)
            logging.info(f"[RESULT] Status: {response.status_code}")
            logging.info(f"[RESPONSE] {response.text}")


if __name__ == "__main__":
    run_all_tests()
