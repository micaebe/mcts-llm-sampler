"""
Dataset sampler.

This script samples data from the GSM8K dataset and sends it to the MCTS sampler channels.
"""

import json
import redis
from datasets import load_dataset


def main():
    """
    Main function to sample data from the GSM8K dataset and send it to the MCTS sampler channels.

    Waits for sample requests from the MCTS sampler and sends samples to the corresponding channels.
    """
    ds = load_dataset("openai/gsm8k", "main")
    redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
    curr_idx = 1000
    # wait for sample requests
    while True:
        _, message = redis_client.brpop("sample_request", timeout=0)
        if message:
            sample_request = json.loads(message)
            channel_id = sample_request["channel_id"]
            sample = ds["train"][curr_idx]
            curr_idx += 1

            sample["channel_id"] = channel_id
            sample["index"] = curr_idx
            redis_client.lpush("sample_channel_" + channel_id, json.dumps(sample))
            print(f"Sample sent: {curr_idx}")
        if curr_idx >= len(ds["train"]):
            raise Exception("All samples have been sent")
        # time.sleep(0.1)


if __name__ == "__main__":
    main()
