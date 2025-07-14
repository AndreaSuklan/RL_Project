An **Actor-Critic network** is a hybrid reinforcement learning architecture that combines the strengths of both policy-based (Actor) and value-based (Critic) methods. The two components work together to learn more efficiently.


## The Actor
The **Actor** is the **policy** network. Its job is to control the agent by deciding which action to take in a given state.

- **Input**: The current state of the environment.

- **Output**: A probability distribution over the possible actions.

- **Goal**: To learn the optimal policy that maximizes rewards.

## The Critic
The **Critic** is the **value** network. Its job is to evaluate the actions taken by the Actor by estimating the value of the state the agent is in.

- **Input**: The current state of the environment.

- **Output**: A single scalar value representing the expected future reward from that state $(V(s))$.

- **Goal**: To accurately predict the value function.

## How They Work Together
The Actor and Critic have a synergistic relationship that improves the learning process:

1. The Actor observes a state and chooses an action.

2. The environment provides a reward and a new state.

3. The Critic evaluates the new state and calculates the "advantage" (it determines if the action taken was better or worse than expected).

4. The Actor uses this feedback from the Critic to update its policy. If the action was good (positive advantage), it adjusts its weights to make that action more likely in the future. If it was bad (negative advantage), it makes it less likely.

This setup stabilizes training because the Actor receives more nuanced feedback than just the raw reward; it learns from a critique of its performance relative to an expected outcome.


Fuentes

