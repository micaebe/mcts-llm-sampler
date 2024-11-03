"""
File writer.

This script writes data to a file from a Redis channel.
"""

import json
import logging
import time

import redis

logging.basicConfig(level=logging.INFO)

file_path = "data.jsonl"


def main():
    """
    Writes data to a file from a Redis channel.
    """
    redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)

    print("Waiting for messages to write to file")

    while True:
        _, message = redis_client.brpop("output_channel", timeout=0)

        if message:
            data = json.loads(message)
            with open(file_path, "a") as f:
                f.write(json.dumps(data) + "\n")
            logging.info("Data written to file")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
