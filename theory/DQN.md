Deep Q-Network (from [Mnih et al. 2013, 2015](https://www.nature.com/articles/nature14236))

In order to fully understand Deep Q-Networks, we first have to present *Q-Learning* and *Semi-Gradient methods*
## Q-Learning
Q-Learning is an *off-policy* control algorithm: unlike SARSA, which updates The action-value function Q based on the observed $S, A$, values at each step, Q-Learning keeps $max_a Q(S', a)$ as a target, thus using effectively 2 different objects for policy update (Hence it's *off-policy*). 

## Semi-Gradient Methods
They belong to approximate methods, where our value-functions $v_{\pi}(s)$ or action-value function $q_{\pi}(s)$ are not just values in a table anymore but parametrized functions e.g $\hat q_{\pi}(s, **w**)$, $\hat v_{\pi}(s, **w**)$.
This is particularly useful in environments where the number of states/actions is extremely large and thus difficut to track in tabular form.

Specifically, it is considered when we are dealing with bootstrapped updates for our weight vector **w**.
We would like to have:
$$
w_{t + 1} = w_t + \alpha[v_{\pi} - \hat v(S_t, w_t)]\nabla \hat(S_t, w_t)
$$

but we often have an imperfect approximation of $v_{\pi} \approx U_t$ and thus can only update

$$
w_{t + 1} = w_t + \alpha[U_t - \hat v(S_t, w_t)]\nabla \hat(S_t, w_t)
$$

But when our target is itself a bootstrapped value, we cannot update it like this because the target itself would be dependent on w.
In order to do this, we resort to **semi-gradient methods**: when making the update $w_{t + 1} \leftarrow w + \alpha[R + \gamma \max_{a}\hat q(S', w) - \hat q(S, w)] \nabla \hat v(S, w)$      
<!-- (comparing the 2 algos in the book, in gradient monte carlo target=$G_t$, while in TD(0) semi-gradient target = $R + \gamma \hat v(S, w)$. -->

## Deep Q-Networks
The key idea behind Deep Q-Networks is to use Artificial Neural Networks as function approximator for the action-value function $\hat q$ updating its values based on the Neural Network's output.

Note that this algorithm is model-free: it solves the reinforcement learning task directly using samples from the game, without explicitly estimating the reward and transition dynamics $P(r,s' \mid s, a)$

It is also off-policy: it learns about the greedy policy $a = argmax_a Q(s,a \mid \theta)$ while following a behaviour distribution that ensures adequate exploration of the state space. In practice, the behaviour distribution is often selected by an $\epsilon$-greedy policy that follows the greedy policy with probability $1 - \epsilon$ and selects a random action with probability $\epsilon$. 

Although the initial example of this architecture was a Deep Convolutional Network that learned to play different ATARI games always using the same underlying architecture, our use case is slightly simpler, as the input is a normal 19-element vector and not images like in the original paper.  
We therefore opted for a simple feedforward neural network with 2 layers (!!!initial simple choice, to be modified if not kept!!!) 

<!-- Key ideas: -->
<!-- - Use deep convolutional ANNs as function approximators -->
<!-- - uses (a semi-gradient form of) Q-Learning. --> 
<!--     - why? Going back to the most famous use (ATARI gameplay), it was more complicated to keep track of afterstates (positions/situations one "ends up to" in a game when doing a move) than in a boardgame like GO. -->  
<!--       TD methods used for go are on-policy, requiring the immediate TD(0) or immediately successive TD(1) step to actually be evaluated. -->  
<!--       Q-learning -->
