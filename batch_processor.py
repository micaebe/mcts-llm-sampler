"""
Batch processor.

This script processes batches of logits and rollouts for the MCTS sampler.
"""

import json
import time

import numpy as np
import redis
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# model = AutoModelForCausalLM.from_config(config)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model.to("cuda")


def logits_batch_processor(states: list[list[int]]) -> np.ndarray:
    """
    Processes a batch of logits.

    Args:
        states: List of states (list of token id lists)
    """
    print(f"Logits batch processor: {len(states)}")
    texts = tokenizer.batch_decode(states)
    tokenized = tokenizer(texts, padding="longest", return_tensors="pt")
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    input_ids = input_ids.to("cpu")
    attention_mask = attention_mask.to("cpu")
    outputs.logits = outputs.logits.to("cpu")
    logits = outputs.logits[:, -1, :].detach().cpu().numpy()
    return logits


def rollout_batch_processor(
    states: list[list[int]], ks: list[int] = []
) -> tuple[list[str], list[list[int]]]:
    """
    Processes a batch of rollouts.

    Args:
        states: List of states
        ks: List of k values (can be empty)
    """
    print(f"Rollout batch processor: {len(states)}")
    texts = tokenizer.batch_decode(states)
    tokenized = tokenizer(texts, padding="longest", return_tensors="pt")
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            do_sample=False,
            num_beams=3,
        )
    generated_solutions = tokenizer.batch_decode(output, skip_special_tokens=True)
    input_ids = input_ids.to("cpu")
    attention_mask = attention_mask.to("cpu")
    output = output.to("cpu")
    first_k_tokens = []
    if len(ks) > 0:
        for i in range(len(output)):
            # find first non padding token
            input_length = len(input_ids[i])
            first_top_k = output[i][input_length : input_length + ks[i]]
            first_top_k = first_top_k.detach().cpu().numpy().tolist()
            first_top_k = [int(x) for x in first_top_k]
            first_k_tokens.append(first_top_k)
    return generated_solutions, first_k_tokens


"""
expected format:

logits_processor:
{
    'state': [1, 2, 3, 4],
    'channel_id': '1234'
}

rollout_processor:
{
    'state': [1, 2, 3, 4],
    'k': 10,
    'channel_id': '1234'
}
"""


def main():
    """
    Main function to process batches of logits and rollouts.

    Waits for rollout and logits requests from the MCTS samplers and processes them in batches.
    Afterwards it sends the processed logits and rollouts to the MCTS sampler channels.
    """
    redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)

    logits_batch: dict[str, list[int]] = {}
    rollout_batch: dict[str, tuple[list[int], int]] = {}

    last_logits_time = time.time()
    last_rollout_time = time.time()

    batch_size = 56
    timeout = 1.0

    while True:
        logits_message = redis_client.brpop("logits_processor", timeout=0.01)
        rollout_message = redis_client.brpop("rollout_processor", timeout=0.01)

        if logits_message:
            last_logits_time = time.time()
            data = json.loads(logits_message[1])
            states = data["state"]
            channel_id = data["channel_id"]
            if channel_id not in logits_batch:
                logits_batch[channel_id] = states
            else:
                print(f"Channel {channel_id} already in logits_batch")

        if rollout_message:
            last_rollout_time = time.time()
            data = json.loads(rollout_message[1])
            states = data["state"]
            channel_id = data["channel_id"]
            k = data["k"]
            if channel_id not in rollout_batch:
                rollout_batch[channel_id] = (states, k)
            else:
                print(f"Channel {channel_id} already in rollout_batch")

        if len(logits_batch) >= batch_size or (
            time.time() - last_logits_time >= timeout and len(logits_batch) > 0
        ):
            logits = logits_batch_processor(logits_batch.values())
            for i in range(len(logits_batch)):
                channel_id = list(logits_batch.keys())[i]
                message = {
                    "state": logits_batch[channel_id],
                    "logits": logits[i].tolist(),
                }
                redis_client.lpush(f"logits_channel_{channel_id}", json.dumps(message))
            logits_batch = {}

        if len(rollout_batch) >= batch_size or (
            time.time() - last_rollout_time >= timeout and len(rollout_batch) > 0
        ):
            states = []
            ks = []
            for channel_id, (state, k) in rollout_batch.items():
                states.append(state)
                ks.append(k)
            generated_solutions, first_k_tokens = rollout_batch_processor(states, ks)
            for i in range(len(rollout_batch)):
                channel_id = list(rollout_batch.keys())[i]
                message = {
                    "state": states[i],
                    "generated_solution": generated_solutions[i],
                    "first_k_tokens": first_k_tokens[i],
                }
                redis_client.lpush(f"rollout_channel_{channel_id}", json.dumps(message))
            rollout_batch = {}
        time.sleep(0.0001)


if __name__ == "__main__":
    main()
