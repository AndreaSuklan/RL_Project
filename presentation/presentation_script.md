
# Deep-Q Network (Giovanni)
Let's now move to the algorithms we used to learn and solve this problem.
Our first choice were *Deep Q-Networks*, a successful algorithm for tackling RL-problems. 
The basic building block of this algorithm is, of course, Q-Learning, whose algorithm you can see here.
Q-Learning belongs to *model free* algorithms: we don't make any assumptions about our environment's transition probabilites or reward structure, instead learning only through direct experience.
The algorithms for this class have a common component, and the only part that is algorithm-dependent, as we'll see with another example later, is how the *TD-Error* error and the *Eligibility Traces* are computed.

In Vanilla Q-Learning, we are in an off-policy setting: that means we have 2 separate policies: One to estimate ( second term in the difference), another one to choose the action greedily (max term). 
Taking the maximum of the policy lead to relatively aggressive updating of the Q-values, with less exploration and suboptimal results in some situations (e.g Cliff Walking example).
Rather than computing the full expectations in the above gradient, it is often
computationally expedient to optimize the loss function by stochastic gradient
descent. The familiar Q-learning algorithm19 can be recovered in this framework
by updating the weights after every time step, replacing the expectations using
single samples, and setting h{
i ~hi{1 .
DQN (also called Fitted Q-Networks) is really just a revisitation of Q-Learning  $q[s_t, a_t] \rightarrow q[s_t, a_t, \phi]$ where we go from a tabular representation of the q-values to one parametrized by a ML model. Differently from Vanilla Q-Learning, this model belongs to the family of *Semi-Gradient Methods*(Sutton and Barto, pag.437, end).
What this means is that instead of a tabular version of the action-value function $q$ we have a parametric approximation of it, like specified here.
This leads, in turn, to the following internal update for weights **w**. Here we update the parameters of our policy instead of the Q-values directly.
The target parameters $w-$ are not updated at every iteration in order not to incur into a *moving target problem*: this is what happens when we are using Q to estimate future rewards, but the Q object is being updated at the same time. Instead they are sampled uniformly at random from the ReplayBuffer, which is extended at each iteration. This helps massively with stabilizing the Neural Network's results.    

Note that this algorithm is model-free: it solves the reinforcement learning task
directly using samples from the emulator, without explicitly estimating the reward
and transition dynamics Pðr,s0 Ds,aÞ. It is also off-policy: it learns about the greedy
policy a~argmaxa0 Qðs,a0 ; hÞ, while following a behaviour distribution that ensures
adequate exploration of the state space. In practice, the behaviour distribution is
often selected by an e-greedy policy that follows the greedy policy with probability
1 2 e and selects a random action with probability e.

