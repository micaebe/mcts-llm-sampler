"""
MCTS sampler.

This script runs the MCTS algorithm on samples from the GSM8K dataset.
"""

import json
import time

import numpy as np
import redis
from transformers import AutoTokenizer

from mcts import MCTS, Node


def parse_numeric_answer(answer_text):
    """
    Extracts a numeric answer from the generated text.
    """
    import re

    match = re.search(r"\boxed\{([\d\.]+)\}", answer_text)
    if match:
        return float(match.group(1))
    else:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_text)
        return float(numbers[-1]) if numbers else None


def get_child_states(node):
    """
    Gets the child states of the given node.

    Args:
        node: Current node
    """
    child_states = []
    for action, child_node in node.children.items():
        state = child_node.state
        Nrsa = node.Nrsa.get(action, 0)
        Wrsa = node.Wrsa.get(action, 0)
        child_states.append(
            {"state": state, "score": Wrsa / Nrsa if Nrsa > 0 else 0, "N": Nrsa}
        )
        child_states.extend(get_child_states(child_node))
    return child_states


def main():
    """
    Main function to run the MCTS sampler.

    Creates a unique channel ID. Then it requests a sample from the sample_request channel.
    Once a sample is received, it runs the MCTS algorithm for a random number of iterations.
    After the MCTS algorithm is finished, it collects the MCTS states and values and sends them to the output channel.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
    sample = None
    channel_id = str(int(time.time() * 1e6))[4:]

    print(f"STARTED MCTS SAMPLER: {channel_id} ---------------------------\n")


    def state_transition_fn(state, action):
        """
        Performs a state transition from the given state and action.

        Args:
            state: Current state (list of token ids)
            action: Action to take (token id)
        """
        new_state = state + [int(action)]
        return new_state


    def value_fn(state):
        # for sampling the first time, we don't have a value network
        return 0


    def rollout_policy_fn(state, solution):
        """
        Performs a rollout from the given state and returns the outcome z(s).

        Args:
            state: Current state
            solution: Target value for the problem (solution text)
        """
        if state[-1] != tokenizer.eos_token_id:
            rollout_request = {"state": state, "k": 0, "channel_id": channel_id}
            redis_client.lpush("rollout_processor", json.dumps(rollout_request))
            _, rollout_message = redis_client.brpop(
                f"rollout_channel_{channel_id}", timeout=0
            )
            generated_solution = json.loads(rollout_message)["generated_solution"]
            print(f"{channel_id[-3:]} --- Rollout received")
        else:
            generated_solution = tokenizer.decode(state, skip_special_tokens=True)
        try:
            generated_answer = parse_numeric_answer(generated_solution)
            expected_answer = parse_numeric_answer(solution)
        except:
            generated_answer = None
            expected_answer = None
        print(
            f"{channel_id[-3:]} --- Rollout:  {generated_answer}, Expected: {expected_answer}"
        )
        if generated_answer is not None and generated_answer == expected_answer:
            return 1
        return -1


    def rollout_policy_fn_k(state, solution, k):
        """
        Performs a rollout from the given state and returns the outcome z(s) as well as the first k tokens.

        This is used to make sampling more efficient, as it "simulates" a deeper tree, so the tree starts at a deeper level.

        Args:
            state: Current state
            solution: Target value for the problem (solution text)
        """
        rollout_request = {"state": state, "k": k, "channel_id": channel_id}
        redis_client.lpush("rollout_processor", json.dumps(rollout_request))
        _, rollout_message = redis_client.brpop(
            f"rollout_channel_{channel_id}", timeout=0
        )
        generated_solution = json.loads(rollout_message)["generated_solution"]
        first_k_tokens = json.loads(rollout_message)["first_k_tokens"]
        try:
            generated_answer = parse_numeric_answer(generated_solution)
            expected_answer = parse_numeric_answer(solution)
        except:
            generated_answer = None
            expected_answer = None
        if generated_answer is not None and generated_answer == expected_answer:
            return 1, first_k_tokens
        return -1, first_k_tokens

    def logits_vector_fn(state):
        """
        Returns the logits for the given state.

        Args:
            state: Current state
        """
        if state[-1] != tokenizer.eos_token_id:
            logits_request = {"state": state, "channel_id": channel_id}
            redis_client.lpush("logits_processor", json.dumps(logits_request))
            _, logits_message = redis_client.brpop(
                f"logits_channel_{channel_id}", timeout=0
            )
            logits = json.loads(logits_message)["logits"]
            print(f"{channel_id[-3:]} --- Logits received")
        else:
            logits = None
        return logits


    # Main loop
    while True:
        if sample is None:
            # request a sample from the sample_request channel
            redis_client.lpush("sample_request", json.dumps({"channel_id": channel_id}))
            _, sample_message = redis_client.brpop(
                f"sample_channel_{channel_id}", timeout=0
            )
            sample = json.loads(sample_message)
            print(f"{channel_id[-3:]} --- Sample received: {sample['index']}")
        else:
            question = sample["question"]
            expected_answer = sample["answer"]
            index = sample["index"]
            root_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Think step by step to solve the problem.",
                },
                {"role": "user", "content": f"{question}"},
            ]

            tokenized_chat = tokenizer.apply_chat_template(
                root_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            root_state = tokenized_chat[0].tolist()

            k = 0
            if np.random.randint(0, 10) < 6:
                k = np.random.randint(1, 15)
            rollout_reward, first_k_tokens = rollout_policy_fn_k(
                root_state, expected_answer, k
            )
            print(
                f"{channel_id[-3:]} --- Rollout reward: {rollout_reward}, First k tokens: {first_k_tokens}"
            )
            if rollout_reward == 1:
                sample = None
                continue
            if k > 0:
                root_state = root_state + first_k_tokens
                print(f"{channel_id[-3:]} --- Root state: {root_state}")
                print(f"{channel_id[-3:]} --- First k tokens: {first_k_tokens}")
            root_node = Node(state=root_state)
            mcts = MCTS(
                logits_vector_fn=logits_vector_fn,
                state_transition_fn=state_transition_fn,
                value_fn=value_fn,
                rollout_policy_fn=rollout_policy_fn,
                solution=expected_answer,
                k=5,
                lmbda=0.5,
                c_puct=2.0,
            )

            num_simulations = np.random.randint(50, 120)
            for iteration in range(num_simulations):
                mcts.run_simulation(root_node)
                if iteration >= 20:
                    if mcts.correct_count >= 15:
                        break

            value_samples = get_child_states(root_node)
            samples_to_append = {
                "id": index,
                "samples": value_samples,
            }

            # send the samples to the output channel
            redis_client.lpush("output_channel", json.dumps(samples_to_append))
            sample = None
        time.sleep(0.01)


if __name__ == "__main__":
    main()
