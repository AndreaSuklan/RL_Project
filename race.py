import gymnasium as gym
from gymnasium import spaces
import pygame
import Box2D
from Box2D import (b2ContactListener, b2World, b2PolygonShape, b2CircleShape, b2FixtureDef, b2RayCastCallback)
from stable_baselines3 import PPO
import numpy as np
import math

# --- Constants ---
# Physics and world properties
FPS = 60  # Frames per second for simulation and rendering
SCALE = 30.0  # Scale for converting between physics units and pixels
VIEWPORT_W = 1200  # Width of the game window in pixels
VIEWPORT_H = 800  # Height of the game window in pixels

# Terrain properties
TERRAIN_STEP = 20 / SCALE  # The maximum change in height between terrain segments
TERRAIN_LENGTH = 200  # Number of segments in the terrain
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4  # Initial height of the terrain
TERRAIN_STARTPAD = 40 / SCALE  # Length of the initial flat area

# Colors
BACKGROUND_COLOR = (135, 206, 235)  # Sky blue
TERRAIN_COLOR = (100, 70, 30)  # Brown


class RayCastCallback(b2RayCastCallback):
    """
    Custom callback for Box2D's RayCast.
    Used for the car's LIDAR sensor to detect the terrain.
    """
    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.point = None
        self.normal = None
        self.fraction = 1.0  # Default to 1.0 (no hit)

    def ReportFixture(self, fixture, point, normal, fraction):
        """
        Called by Box2D for each fixture intersected by the ray.
        Stores the details of the closest hit.
        """
        self.fixture = fixture
        self.point = point
        self.normal = normal
        self.fraction = fraction
        return fraction


class MyContactListener(b2ContactListener):
    """
    Custom contact listener to detect collisions.
    Specifically used to end the episode if the human driver hits the terrain.
    """
    def __init__(self, env):
        super(MyContactListener, self).__init__()
        self.env = env

    def BeginContact(self, contact):
        """
        Called when two fixtures begin to touch.
        """
        dataA = contact.fixtureA.body.userData
        dataB = contact.fixtureB.body.userData

        # Check if the human has hit the terrain
        if (dataA == "human" and dataB == "terrain") or \
           (dataA == "terrain" and dataB == "human"):
            self.env.terminated = True


class HillClimbEnv(gym.Env):
    """
    A custom Gymnasium environment for a Hill Climb Racing-like game.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.b2World = None
        self.car = None
        self.terrain_poly = []
        self.terrain = None
        self.terminated = False
        self.contact_listener = None

        # Action space: 0=do nothing, 1=accelerate left, 2=accelerate right
        self.action_space = spaces.Discrete(3)

        # Observation space: a continuous vector of sensor readings
        high = np.array([np.inf] * 18, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _destroy(self):
        """
        Destroys all Box2D bodies and resets the environment state.
        """
        if not self.b2World: return
        for body in self.b2World.bodies: self.b2World.DestroyBody(body)
        self.terrain_poly = []
        self.car = None
        self.terrain = None
        self.b2World = None
        self.contact_listener = None

    def _create_terrain(self):
        """
        Generates the random, hilly terrain for the car to drive on.
        """
        y = TERRAIN_HEIGHT
        x = 0
        self.terrain_poly = []
        self.terrain_poly.append((x, TERRAIN_HEIGHT))
        x += TERRAIN_STARTPAD
        self.terrain_poly.append((x, TERRAIN_HEIGHT))

        for _ in range(TERRAIN_LENGTH):
            y += self.np_random.uniform(-TERRAIN_STEP * 2, TERRAIN_STEP * 2)
            x += self.np_random.uniform(TERRAIN_STEP * 2, TERRAIN_STEP * 4)
            self.terrain_poly.append((x, y))

        self.terrain = self.b2World.CreateStaticBody(userData="terrain")
        self.terrain.CreateEdgeChain(self.terrain_poly)

    def _create_car(self):
        """
        Creates the car, including the chassis, wheels, suspensions, and driver.
        """
        chassis_body = self.b2World.CreateDynamicBody(
            position=(TERRAIN_STARTPAD / 2, TERRAIN_HEIGHT + 1),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[(-1, -0.2), (1, -0.2), (1, 0.2), (-1, 0.2)]),
                density=1.0, filter=Box2D.b2Filter(groupIndex=-1)
            )
        )
        chassis_body.userData = "chassis"

        wheels = []
        for i in [-1, 1]:
            wheel = self.b2World.CreateDynamicBody(
                position=(chassis_body.position.x + i * 0.8, chassis_body.position.y - 0.4),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=0.4),
                    density=1.0, restitution=0.2, friction=1,
                    filter=Box2D.b2Filter(groupIndex=-1)
                )
            )
            wheel.userData = f"wheel_{i}"
            wheels.append(wheel)

        suspensions = []
        for i in range(2):
            joint = self.b2World.CreateWheelJoint(
                bodyA=chassis_body, bodyB=wheels[i], anchor=wheels[i].position,
                axis=(0, 1), motorSpeed=0, maxMotorTorque=20, enableMotor=True,
                frequencyHz=4.0, dampingRatio=0.7
            )
            suspensions.append(joint)

        human = self.b2World.CreateDynamicBody(
            position=(chassis_body.position.x, chassis_body.position.y + 1),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[(-0.3, 0.35), (0.3, 0.35), (0.3, -0.35), (-0.3, -0.35)]),
                density=1.0, filter=Box2D.b2Filter(groupIndex=-1)
            )
        )
        human.userData = "human"

        seat = self.b2World.CreateWheelJoint(
            bodyA=chassis_body, bodyB=human, anchor=human.position,
            axis=(0, 1), motorSpeed=0, maxMotorTorque=0, enableMotor=False,
            frequencyHz=4.0, dampingRatio=0.7
        )

        self.car = (chassis_body, wheels, suspensions, human, seat)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new episode.
        """
        super().reset(seed=seed)
        self._destroy()

        self.terminated = False

        self.b2World = b2World(gravity=(0, -9.8))
        self.contact_listener = MyContactListener(self)
        self.b2World.contactListener = self.contact_listener

        self._create_terrain()
        self._create_car()
        self.prev_shaping = None
        self.step_count = 0
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        """
        Advances the environment by one timestep.
        """
        self.step_count += 1
        chassis, wheels, suspensions, human, seat = self.car
        motor_speed = 0
        if action == 1:
            motor_speed -= 15.0
        elif action == 2:
            motor_speed += 15.0
        for suspension in suspensions:
            suspension.motorSpeed = motor_speed

        self.b2World.Step(1.0 / FPS, 6 * 30, 2 * 30)
        obs = self._get_observation()

        # Reward shaping: encourage forward movement
        reward = 0
        shaping = chassis.position.x
        if self.prev_shaping is not None:
            reward = 10 * (shaping - self.prev_shaping)
        self.prev_shaping = shaping

        # Penalize for going backwards
        if chassis.position.x < TERRAIN_STARTPAD / 2:
            self.terminated = True
            reward = -200

        # Truncate the episode if it runs for too long
        truncated = self.step_count > 2000

        # Penalize for termination (e.g., crashing)
        if self.terminated:
            reward = -100.0

        if self.render_mode == "human":
            self.render()

        return obs, reward, self.terminated, truncated, {}

    def _get_observation(self):
        """
        Gathers the current state of the environment for the agent.
        """
        chassis, wheels, suspensions, human, seat = self.car
        pos, vel = chassis.position, chassis.linearVelocity
        state = [
            chassis.angle, chassis.angularVelocity, vel.x, vel.y,
            wheels[0].angularVelocity, wheels[1].angularVelocity,
            suspensions[0].translation, suspensions[1].translation
        ]

        # LIDAR sensor readings
        lidar_readings = []
        for angle in np.linspace(-math.pi / 2, math.pi / 2, 10):
            p1 = pos
            p2 = (pos.x + math.cos(chassis.angle + angle) * 50,
                  pos.y + math.sin(chassis.angle + angle) * 50)

            callback = RayCastCallback()
            self.b2World.RayCast(callback, p1, p2)
            lidar_readings.append(callback.fraction)

        state.extend(lidar_readings)
        return np.array(state, dtype=np.float32)

    def render(self):
        """
        Renders the current state of the environment using Pygame.
        """
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            pygame.display.set_caption("Hill Climb RL")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.b2World is None: return

        pygame.event.pump()
        self.screen.fill(BACKGROUND_COLOR)

        # Scroll the view to follow the car
        scroll = self.car[0].position.x * SCALE - VIEWPORT_W / 4
        vertices = [(v[0] * SCALE - scroll, VIEWPORT_H - v[1] * SCALE) for v in self.terrain_poly]
        pygame.draw.lines(self.screen, TERRAIN_COLOR, False, vertices, 3)

        # Draw the car and its components
        chassis, wheels, _, human, _ = self.car
        for body in [chassis] + wheels + [human]:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    poly_vertices = [(body.transform * v) * SCALE for v in shape.vertices]
                    poly_vertices = [(v[0] - scroll, VIEWPORT_H - v[1]) for v in poly_vertices]
                    pygame.draw.polygon(self.screen, (200, 50, 50), poly_vertices)
                elif isinstance(shape, b2CircleShape):
                    pos = (body.position * SCALE)
                    pos = (pos[0] - scroll, VIEWPORT_H - pos[1])
                    pygame.draw.circle(self.screen, (50, 50, 50), pos, shape.radius * SCALE)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """
        Closes the Pygame window and quits Pygame.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


if __name__ == '__main__':
    from stable_baselines3 import PPO

    # Create the environment
    env = HillClimbEnv()

    # Instantiate the PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_hcr_tensorboard/")

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=20000)
    model.save("ppo_hcr_model")
    print("Training finished and model saved!")

    # Clean up the old model and environment
    del model
    env.close()

    # Load the trained model
    model = PPO.load("ppo_hcr_model")

    # Create a new environment for rendering
    env = HillClimbEnv(render_mode="human")
    obs, info = env.reset()

    # Run the trained agent for a fixed number of steps
    for _ in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()