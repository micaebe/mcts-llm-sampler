import numpy as np


class Node:
    """
    Represents a node in the MCTS tree with per-action statistics.
    """

    def __init__(self, state, parent=None):
        self.state = state  # The state represented by this node (as a string)
        self.parent = parent  # Parent node
        self.children = {}  # Map from action to child node
        self.Psa = {}  # Prior probabilities P(s,a) per action
        self.Nvsa = {}  # Visit counts Nv(s,a) per action (from value network)
        self.Nrsa = {}  # Visit counts Nr(s,a) per action (from rollouts)
        self.Wvsa = {}  # Total value Wv(s,a) per action (from value network)
        self.Wrsa = {}  # Total value Wr(s,a) per action (from rollouts)
        self.Qsa = {}  # Combined mean action value Q(s,a)
        self.is_expanded = False  # Flag indicating if the node has been expanded


class MCTS:
    """
    Monte Carlo Tree Search implementation aligned with the AlphaGo algorithm.
    """

    def __init__(
        self,
        logits_vector_fn,
        state_transition_fn,
        value_fn,
        rollout_policy_fn,
        solution: str,
        k=10,
        lmbda=0.5,
        c_puct=1.0,
    ):
        """
        Args:
            logits_vector_fn: Function to get logits vector for a state
            state_transition_fn: Function to get next state given a state and action
            value_fn: Value network function v(s)
            rollout_policy_fn: Rollout policy function z(s)
            solution: Target value for the problem (solution text)
            k: Number of top actions to consider
            lmbda: Mixing parameter between value network and rollout policy
            c_puct: Exploration constant
        """
        self.logits_vector_fn = logits_vector_fn
        self.state_transition_fn = state_transition_fn
        self.value_fn = value_fn
        self.rollout_policy_fn = rollout_policy_fn
        self.solution = solution
        self.k = k
        self.lmbda = lmbda
        self.c_puct = c_puct
        self.correct_count = 0

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def run_simulation(self, root):
        """
        Runs one simulation of MCTS starting from the root node.

        Args:
            root: Root node to start the simulation from
        """
        node = root
        path = []
        while True:
            if not node.is_expanded:
                # Expand the node
                self.expand(node)
                break
            else:
                # Select the best action
                action = self.select(node)
                path.append((node, action))
                if action in node.children:
                    node = node.children[action]
                else:
                    # Create a new child node
                    new_state = self.state_transition_fn(node.state, action)
                    child_node = Node(new_state, parent=node)
                    node.children[action] = child_node
                    node = child_node

        # Evaluate the leaf node
        value_v, value_r = self.evaluate(node)
        # Backpropagate the values
        self.backpropagate(path, value_v, value_r)

    def select(self, node):
        """
        Selects an action using the PUCT formula.

        Args:
            node: Current node
        """
        total_Nr = sum(node.Nrsa.get(a, 0) for a in node.Psa.keys())
        total_Nr = max(total_Nr, 1)  # Avoid division by zero

        best_uct = -float("inf")
        best_action = None
        for action in node.Psa.keys():
            Psa = node.Psa[action]
            Nrsa = node.Nrsa.get(action, 0)
            Qsa = node.Qsa[action]

            # Compute u(s,a)
            u_sa = self.c_puct * Psa * np.sqrt(total_Nr) / (1 + Nrsa)

            uct_value = Qsa + u_sa
            if uct_value > best_uct:
                best_uct = uct_value
                best_action = action

        return best_action

    def expand(self, node):
        """
        Expands a node by processing it with the policy network to get P(s,a).

        Args:
            node: Current node
        """
        logits_vector = self.logits_vector_fn(node.state)
        # Convert logits to probabilities
        probs = self.softmax(logits_vector)
        # Select top k actions
        top_k_indices = np.argsort(probs)[-self.k :]
        for idx in top_k_indices:
            action = idx
            Psa = probs[idx]
            node.Psa[action] = Psa
            # Initialize counts and values
            node.Nvsa[action] = 0
            node.Nrsa[action] = 0
            node.Wvsa[action] = 0
            node.Wrsa[action] = 0
            node.Qsa[action] = 0
        node.is_expanded = True

    def evaluate(self, node):
        """
        Evaluates the leaf node using the value network and rollout policy.

        Returns:
            value_v: Value from the value network
            value_r: Outcome from the rollout
        """
        value_v = self.value_fn(node.state)
        value_r = self.rollout_policy_fn(node.state, self.solution)
        if value_r == 1:
            self.correct_count += 1
        return value_v, value_r

    def backpropagate(self, path, value_v, value_r):
        """
        Backpropagates the simulation results up the tree.

        Args:
            path: List of (node, action) pairs traversed during the simulation.
            value_v: Value from the value network
            value_r: Outcome of the rollout z
        """
        for node, action in reversed(path):
            # Update counts and values from value network
            node.Nvsa[action] = node.Nvsa.get(action, 0) + 1
            node.Wvsa[action] = node.Wvsa.get(action, 0) + value_v

            # Update counts and values from rollout
            node.Nrsa[action] = node.Nrsa.get(action, 0) + 1
            node.Wrsa[action] = node.Wrsa.get(action, 0) + value_r

            # Recalculate Q(s,a)
            Nvsa = node.Nvsa[action]
            Nrsa = node.Nrsa[action]
            Wvsa = node.Wvsa[action]
            Wrsa = node.Wrsa[action]
            qv = Wvsa / Nvsa if Nvsa > 0 else 0
            qr = Wrsa / Nrsa if Nrsa > 0 else 0
            node.Qsa[action] = (1 - self.lmbda) * qv + self.lmbda * qr
