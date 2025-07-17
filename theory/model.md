# Problem Definition
The problem is to train an **agent** (the car) to navigate a procedurally generated, 2D physics-based **environment**. The agent's goal is to learn an optimal **policy** ($\pi$) that maps **states** to **actions** in order to maximize its cumulative **reward** by driving as far as possible while collecting coins and avoiding crashes.

## Environment
The environment is a 2D world simulated by the Box2D physics engine, characterized by:

- **Hilly Terrain**: A randomly generated, continuous track with varying slopes.
- **Gravity**: A constant downward force affecting all dynamic objects.
- **Coins**: Collectible items placed along the terrain that provide a positive reward.

## Agent
The agent is the vehicle controlled by the RL algorithm. It consists of:

- A rectangular **chassis**.
- Two independently powered **wheels** connected to the chassis by suspensions.
- A **driver** body attached to the chassis. A collision between the driver's head and the terrain ends the episode.

## State Space
The state is a vector of 19 continuous values that provides the agent with a snapshot of the environment at each timestep. It includes:

- **Car Physics** (7 values):

    1. Chassis Angle

    2. Chassis Angular Velocity

    3. Chassis Linear Velocity (X-axis)

    4. Chassis Linear Velocity (Y-axis)

    5. Front Suspension Translation

    6. Rear Suspension Translation

    7. Airborne Status (1 if airborne, 0 if on the ground)

- **LIDAR Sensor Data** (10 values):

    Ten distance readings from a forward-facing LIDAR array, measuring the distance to the terrain. This helps the agent "see" the upcoming hills.

- **Nearest Coin Information** (2 values):

    The relative X and Y coordinates of the nearest coin, transformed into the car's local reference frame. This tells the agent where to go to get the next coin.

## Action Space
The action space is discrete, consisting of 4 possible actions the agent can take at each step:

- Action 0: Do Nothing (coast)

- Action 1: Accelerate Left / Rotate Forward (while airborne)

- Action 2: Accelerate Right / Rotate Backward (while airborne)


## Reward Function
The reward function is designed to guide the agent toward the desired behavior:

- **Positive Rewards**:

    - $+X$ points for each coin collected.

    - A small positive reward proportional to the forward distance traveled since the last step.

- **Negative Rewards**:

    - $-X$ points for terminating the episode (i.e., the driver's head hits the terrain).

    - A large penalty if the car moves backward behind the starting line.