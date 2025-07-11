import gymnasium as gym
from gymnasium import spaces
import pygame
import Box2D
from Box2D import (b2ContactListener, b2World, b2PolygonShape, b2CircleShape, b2FixtureDef, b2RayCastCallback) 
from stable_baselines3 import PPO
import numpy as np
import math

#   Constants
FPS = 60
SCALE = 30.0
VIEWPORT_W = 1200
VIEWPORT_H = 800
TERRAIN_STEP = 20 / SCALE
TERRAIN_LENGTH = 200
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_STARTPAD = 40 / SCALE
ACCELERATION = 15
TORQUE = 10

#   Colors  
BACKGROUND_COLOR = (135, 206, 235)
TERRAIN_COLOR = (100, 70, 30)


class RayCastCallback(b2RayCastCallback):
    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.point = None
        self.normal = None
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.point = point
        self.normal = normal
        self.fraction = fraction
        return fraction

class MyContactListener(b2ContactListener):
    def __init__(self, env):
        super(MyContactListener, self).__init__()
        self.env = env
            
    def BeginContact(self, contact):
        dataA = contact.fixtureA.body.userData
        dataB = contact.fixtureB.body.userData

        if (dataA == "human" and dataB == "terrain") or \
           (dataA == "terrain" and dataB == "human"):
            self.env.terminated = True
        
        if (dataA == "terrain" and dataB == "wheels_1") or \
           (dataA == "wheels_1" and dataB == "terrain"):
            self.airborn = False

        elif (dataA == "terrain" and dataB == "wheels_-1") or \
           (dataA == "wheels_-1" and dataB == "terrain"):
            self.airborn = False

        else:
            self.airborn = True
        

class HillClimbEnv(gym.Env):
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
        self.airborn = True
        self.motorspeed = 0.0
        
        self.contact_listener = None

        self.action_space = spaces.Discrete(3)
        high = np.array([np.inf] * 17, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _destroy(self):
        if not self.b2World: return
        for body in self.b2World.bodies: self.b2World.DestroyBody(body)
        self.terrain_poly = []
        self.car = None
        self.terrain = None
        self.b2World = None
        self.contact_listener = None


    def _create_terrain(self):
        y = TERRAIN_HEIGHT
        x = 0
        self.terrain_poly = []
        self.terrain_poly.append((x, TERRAIN_HEIGHT))
        x += TERRAIN_STARTPAD
        self.terrain_poly.append((x, TERRAIN_HEIGHT))
        
        for _ in range(TERRAIN_LENGTH):
            step = self.np_random.uniform(-TERRAIN_STEP*2, TERRAIN_STEP*2)

            while ((y + step) <= 0):
                step = self.np_random.uniform(-TERRAIN_STEP*2, TERRAIN_STEP*2)

            y += step
            x += self.np_random.uniform(TERRAIN_STEP * 2, TERRAIN_STEP * 4)
            self.terrain_poly.append((x, y))

        self.terrain = self.b2World.CreateStaticBody(userData="terrain")
        self.terrain.CreateEdgeChain(self.terrain_poly)

    def _create_car(self):
        chassis_body = self.b2World.CreateDynamicBody(
            position=(TERRAIN_STARTPAD / 2, TERRAIN_HEIGHT + 1),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[(-1,-0.2),(1,-0.2),(1,0.2),(-1,0.2)]),
                density=1.0, filter=Box2D.b2Filter(groupIndex=-1)
            )
        )
        chassis_body.userData = "chassis"

        wheels = []
        for i in [-1, 1]:
            wheel = self.b2World.CreateDynamicBody(
                position=(chassis_body.position.x + i*0.8, chassis_body.position.y - 0.4),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=0.4),
                    density=1.0, restitution=0.2, friction=5,
                    filter=Box2D.b2Filter(groupIndex=-1)
                )
            )
            wheel.userData = f"wheel_{i}"
            wheels.append(wheel)

        suspensions = []
        for i in range(2):
            joint = self.b2World.CreateWheelJoint(
                bodyA=chassis_body, bodyB=wheels[i], anchor=wheels[i].position,
                axis=(0,1), motorSpeed=0, maxMotorTorque=20, enableMotor=True,
                frequencyHz=4.0, dampingRatio=0.7
            )
            suspensions.append(joint)

        human = self.b2World.CreateDynamicBody(
            position=(chassis_body.position.x, chassis_body.position.y + 1),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[(-0.3,0.35),(0.3,0.35),(0.3,-0.35),(-0.3,-0.35)]),
                density=1.0, filter=Box2D.b2Filter(groupIndex=-1)
            )
        )
        human.userData = "human"

        seat = self.b2World.CreateWheelJoint(
                bodyA=chassis_body, bodyB=human, anchor=human.position,
                axis=(0,1), motorSpeed=0, maxMotorTorque=0, enableMotor=False,
                frequencyHz=4.0, dampingRatio=0.7
            )

        self.car = (chassis_body, wheels, suspensions, human, seat)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        
        self.terminated = False
        self.airborn = True
        self.motorspeed = 0.0
        
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
        self.step_count += 1
        chassis, wheels, suspensions, human, seat  = self.car

        if action == 1: 
            self.motorspeed -= ACCELERATION
            if self.airborn == True:
                chassis.ApplyTorque(TORQUE, wake = True)
        elif action == 2: 
            self.motorspeed += ACCELERATION
            if self.airborn == True:
                chassis.ApplyTorque(-TORQUE, wake = True)
        else:
            chassis.ApplyTorque(0.0, wake = True)

        for suspension in suspensions: 
            suspension.motorSpeed = self.motorspeed

        self.b2World.Step(1.0/FPS, 6*30, 2*30)
        obs = self._get_observation()

        reward = 0
        shaping = chassis.position.x
        if self.prev_shaping is not None: reward = 10*(shaping - self.prev_shaping)
        self.prev_shaping = shaping

        # Kill if going backwards
        if chassis.position.x < TERRAIN_STARTPAD/2:

            self.terminated = True
            reward = -200

        truncated = self.step_count > 2000
        
        if self.terminated:
            reward = -100.0

        if self.render_mode == "human": self.render()
        return obs, reward, self.terminated, truncated, {}

    def _get_observation(self):
        chassis, wheels, suspensions, human, seat = self.car
        pos, vel = chassis.position, chassis.linearVelocity
        state = [
            chassis.angle, chassis.angularVelocity, vel.x, vel.y,
            suspensions[0].translation, suspensions[1].translation,
            int(self.airborn)
        ]
        lidar_readings = []
        for angle in np.linspace(-math.pi/2, math.pi/2, 10):
            p1 = pos
            p2 = (pos.x + math.cos(chassis.angle + angle) * 50,
                  pos.y + math.sin(chassis.angle + angle) * 50)
            
            callback = RayCastCallback()
            self.b2World.RayCast(callback, p1, p2)
            lidar_readings.append(callback.fraction)
            
        state.extend(lidar_readings)
        return np.array(state, dtype=np.float32)

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            pygame.display.set_caption("Hill Climb RL")
        if self.clock is None: self.clock = pygame.time.Clock()

        if self.b2World is None: return

        pygame.event.pump()

        self.screen.fill(BACKGROUND_COLOR)
        scroll = self.car[0].position.x * SCALE - VIEWPORT_W / 4
        vertices = [(v[0]*SCALE - scroll, VIEWPORT_H - v[1]*SCALE) for v in self.terrain_poly]
        pygame.draw.lines(self.screen, TERRAIN_COLOR, False, vertices, 3)

        chassis, wheels, _, human, _ = self.car
        for body in [chassis] + wheels + [human]:
            for fixture in body.fixtures:
                shape = fixture.shape
                i = 0
                if isinstance(shape, b2PolygonShape):
                    poly_vertices = [(body.transform * v) * SCALE for v in shape.vertices]
                    poly_vertices = [(v[0]-scroll, VIEWPORT_H - v[1]) for v in poly_vertices]
                    pygame.draw.polygon(self.screen, (200 - i,50,50 + i), poly_vertices)
                    i += 100
                elif isinstance(shape, b2CircleShape):
                    pos = (body.position * SCALE)
                    pos = (pos[0]-scroll, VIEWPORT_H - pos[1])
                    pygame.draw.circle(self.screen, (50,50,50), pos, shape.radius * SCALE)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

if __name__ == '__main__':
    from stable_baselines3 import PPO

    env = HillClimbEnv()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_hcr_tensorboard/")
    print("Starting training...")
    model.learn(total_timesteps=200000)
    model.save("ppo_hcr_model")
    print("Training finished and model saved!")

    del model
    model = PPO.load("ppo_hcr_model")
    
    env = HillClimbEnv(render_mode="human")
    obs, info = env.reset()

    for _ in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()