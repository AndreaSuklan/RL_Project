import gymnasium as gym
from gymnasium import spaces
import pygame
import Box2D
from Box2D import (b2ContactListener, b2World, b2PolygonShape, b2CircleShape, b2FixtureDef, b2RayCastCallback)
import numpy as np
import math
from scipy.interpolate import make_interp_spline

# --- Constants ---
FPS = 60
SCALE = 30.0
VIEWPORT_W = 1200
VIEWPORT_H = 800
TERRAIN_STEP = 20 / SCALE
TERRAIN_LENGTH = 200
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_STARTPAD = 40 / SCALE
ACCELERATION = 20.0
TORQUE = 10.0

# --- Colors ---
BACKGROUND_COLOR = (135, 206, 235)
SOIL_COLOR = (100, 70, 30)
GRASS_COLOR = (20, 150, 30)
COIN_COLOR = (255, 215, 0)

# --- Rewards and Penalties ---
REWARD_DISTANCE = 5.0
REWARD_COIN = 50.0
REWARD_AIR_TIME = 5
PENALTY_COLLISION = -200.0
PENALTY_TIME = -0.1


class RayCastCallback(b2RayCastCallback):
    """Callback for LIDAR raycasts to detect terrain."""
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
    """Detects collisions between game objects."""
    def __init__(self, env):
        super(MyContactListener, self).__init__()
        self.env = env

    def BeginContact(self, contact):
        dataA = contact.fixtureA.body.userData
        dataB = contact.fixtureB.body.userData

        # --- Coin Collection ---
        if "coin" in str(dataB) and dataA in ["chassis", "human", "wheel_-1", "wheel_1"]:
            # Append the user data (a string), NOT the body object
            if dataB not in self.env.coins_to_remove:
                self.env.coins_to_remove.append(dataB)
        
        if "coin" in str(dataA) and dataB in ["chassis", "human", "wheel_-1", "wheel_1"]:
            # Append the user data (a string), NOT the body object
            if dataA not in self.env.coins_to_remove:
                self.env.coins_to_remove.append(dataA)

        # --- Crash Detection ---
        if (dataA == "human" and dataB == "terrain") or \
           (dataA == "terrain" and dataB == "human"):
            self.env.terminated = True

        # --- Air-time Detection ---
        if dataA == "terrain" and dataB in ["chassis", "wheel_-1", "wheel_1"]:
            self.env.ground_contacts.add(dataB)
        elif dataB == "terrain" and dataA in ["chassis", "wheel_-1", "wheel_1"]:
            self.env.ground_contacts.add(dataA)

    def EndContact(self, contact):
        dataA = contact.fixtureA.body.userData
        dataB = contact.fixtureB.body.userData
        
        # --- Air-time Detection ---
        if dataA == "terrain" and dataB in self.env.ground_contacts:
            self.env.ground_contacts.remove(dataB)
        elif dataB == "terrain" and dataA in self.env.ground_contacts:
            self.env.ground_contacts.remove(dataA)


class HillClimbEnv(gym.Env):
    """Custom Gymnasium environment for the Hill Climb game."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, enable_coins=True):
        super().__init__()
        self.render_mode = render_mode
        self.enable_coins = enable_coins
        self.screen = None
        self.clock = None

        # --- Terrain ---
        self.min_height = TERRAIN_HEIGHT / 2
        self.max_height = (VIEWPORT_H / SCALE) * 0.8
        
        # --- World and Game Objects ---
        self.b2World = None
        self.car = None
        self.terrain = None
        self.wall = None
        self.terrain_poly = []
        self.smooth_terrain_poly = []
        self.coins = []
        self.coins_to_remove = []
        self.bodies_to_destroy = []
        self.ground_contacts = set()

        # --- Environment State ---
        self.terminated = False
        self.airborn = True
        self.motorspeed = 0.0
        self.prev_shaping = None
        self.step_count = 0
        self.air_time_steps = 0
        self.current_score = 0.0
        self.coins_collected = 0
        self.max_x_achieved = 0.0
        self.steps_since_progress = 0
        
        # --- Action & Observation Spaces ---
        self.action_space = spaces.Discrete(3)
        obs_size = 19 if self.enable_coins else 17
        high = np.array([np.inf] * obs_size, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _destroy(self):
        if not self.b2World: return
        for body in list(self.b2World.bodies):
            self.b2World.DestroyBody(body)
        self.terrain_poly = []
        self.car = None
        self.terrain = None
        self.b2World = None
        self.coins = []
        self.coins_to_remove = []

    def _create_terrain(self):

        self.b2World = b2World(gravity=(0, -9.8))
        self.b2World.contactListener = MyContactListener(self)
        
        # Generate the initial terrain
        self.terrain_poly = [(0, TERRAIN_HEIGHT)]
        start_pad = [(TERRAIN_STARTPAD, TERRAIN_HEIGHT)]
        initial_chunk = self._generate_terrain_chunk(TERRAIN_STARTPAD, TERRAIN_HEIGHT, TERRAIN_LENGTH)
        self.terrain_poly.extend(start_pad + initial_chunk)
        
        # Create the initial physics body
        self.terrain = self.b2World.CreateStaticBody(userData="terrain")
        self.terrain.CreateEdgeChain(self.terrain_poly)

        # Create a solid wall at the start
        self.wall = self.b2World.CreateStaticBody(userData="wall")
        wall_vertices = [
            (0, 0),                                     # Bottom-right
            (-VIEWPORT_W / SCALE, 0),                   # Bottom-left
            (-VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),  # Top-left
            (0, VIEWPORT_H / SCALE)                     # Top-right
        ]
        self.wall.CreatePolygonFixture(shape=b2PolygonShape(vertices=wall_vertices))
                                                           
        # Re-create the smooth terrain points for rendering
        x_coords = [p[0] for p in self.terrain_poly]
        y_coords = [p[1] for p in self.terrain_poly]
        spline = make_interp_spline(x_coords, y_coords, k=3)
        num_smooth_points = len(self.terrain_poly) * 10
        x_smooth = np.linspace(min(x_coords), max(x_coords), num_smooth_points)
        y_smooth = spline(x_smooth)
        self.smooth_terrain_poly = list(zip(x_smooth, y_smooth))

    def _create_coins(self):
        if not self.enable_coins:
            self.coins = []
            return
        
        self.coins = []
        for i in range(5, len(self.terrain_poly), 5):
            x1, y1 = self.terrain_poly[i - 1]
            x2, y2 = self.terrain_poly[i]
            
            coin_x = (x1 + x2) / 2
            coin_y = (y1 + y2) / 2 + 0.8
            
            coin = self.b2World.CreateStaticBody(
                position=(coin_x, coin_y),
                fixtures=b2FixtureDef(
                    isSensor=True,
                    shape=b2CircleShape(radius=0.3),
                )
            )
            coin.userData = f"coin_{len(self.coins)}"
            self.coins.append(coin)

    def _create_car(self):
        # --- Chassis ---
        # A more detailed shape for the chassis to resemble a jeep
        chassis_vertices = [
            (-1.2, -0.2), (1.3, -0.2), (1.4, 0.1), (1.1, 0.3),
            (-0.6, 0.35), (-1.25, 0.1)
        ]
        chassis_body = self.b2World.CreateDynamicBody(
            position=(TERRAIN_STARTPAD / 2, TERRAIN_HEIGHT + 1),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=chassis_vertices),
                density=1.0, filter=Box2D.b2Filter(groupIndex=-1)
            )
        )
        chassis_body.userData = "chassis"

        # --- Wheels ---
        wheels = []
        for i in [-1, 1]:
            wheel = self.b2World.CreateDynamicBody(
                position=(chassis_body.position.x + i * 0.85, chassis_body.position.y - 0.25),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=0.4),
                    density=1.0, restitution=0.2, friction=5,
                    filter=Box2D.b2Filter(groupIndex=-1)
                )
            )
            wheel.userData = f"wheel_{i}"
            wheels.append(wheel)

        # --- Suspensions ---
        suspensions = []
        for i in range(2):
            joint = self.b2World.CreateWheelJoint(
                bodyA=chassis_body, bodyB=wheels[i], anchor=wheels[i].position,
                axis=(0, 1), motorSpeed=0, maxMotorTorque=20, enableMotor=True,
                frequencyHz=4.0, dampingRatio=0.7
            )
            suspensions.append(joint)

        # --- Driver ---
        human_body = self.b2World.CreateDynamicBody(
            position=(chassis_body.position.x - 0.2, chassis_body.position.y + 0.3),
            fixtures=[
                # The body
                b2FixtureDef(
                    shape=b2PolygonShape(vertices=[(-0.25, -0.3), (0.25, -0.3), (0.25, 0.3), (-0.25, 0.3)]),
                    density=1.0, filter=Box2D.b2Filter(groupIndex=-1)
                ),
                # The head (helmet)
                b2FixtureDef(
                    shape=b2CircleShape(pos=(0, 0.45), radius=0.2),
                    density=1.0, filter=Box2D.b2Filter(groupIndex=-1)
                )
            ]
        )
        human_body.userData = "human"
        
        # Weld the driver to the chassis
        self.b2World.CreateWeldJoint(bodyA=chassis_body, bodyB=human_body, anchor=human_body.position)
        self.car = (chassis_body, wheels, suspensions, human_body)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._destroy()
        
        self.b2World = b2World(gravity=(0, -9.8))
        self.b2World.contactListener = MyContactListener(self)
        
        self._create_terrain()
        self._create_car()
        self._create_coins()

        self.terminated = False
        self.airborn = True
        self.motorspeed = 0.0
        self.coins_to_remove = []
        self.prev_shaping = None
        self.step_count = 0
        self.current_score = 0.0
        self.coins_collected = 0
        self.air_time_steps = 0
        self.max_x_achieved = TERRAIN_STARTPAD / 2
        self.steps_since_progress = 0

        self.bodies_to_destroy.clear()
        self.ground_contacts.clear()
        
        return self._get_observation(), {}


    def _calculate_reward(self):
        """Calculates the reward for the current step."""
        reward = 0
        
        # Coin Reward
        if self.enable_coins and self.coins_to_remove:            # Create a set of user data strings of coins to be removed
            unique_coins_to_remove = set(self.coins_to_remove)
            self.coins_collected += len(unique_coins_to_remove)

            # Create a list to hold the coin bodies we find
            coins_found_for_removal = []
            
            # Iterate through a copy of the master coin list to find the bodies
            for coin in list(self.coins):
                if coin.userData in unique_coins_to_remove:
                    coins_found_for_removal.append(coin)

            # Process the found bodies
            for coin in coins_found_for_removal:
                reward += REWARD_COIN
                if coin in self.coins:
                    self.bodies_to_destroy.append(coin)
                    self.coins.remove(coin)

            self.coins_to_remove.clear()
        
        # Progress Reward
        shaping = self.car[0].position.x
        if self.prev_shaping is not None:
            reward += REWARD_DISTANCE * (shaping - self.prev_shaping)
        self.prev_shaping = shaping

        # Air-time Reward
        if self.airborn and self.air_time_steps > 0 and self.air_time_steps % FPS == 0:
            reward += REWARD_AIR_TIME

        # Crashing Penalty
        if self.terminated:
            reward = PENALTY_COLLISION

        reward += PENALTY_TIME

        return reward
    

    def _generate_terrain_chunk(self, start_x, start_y, num_segments):
        """Generates a new chunk of terrain points."""
        new_poly = []
        x, y = start_x, start_y
        
        for _ in range(num_segments):
            step = self.np_random.uniform(-TERRAIN_STEP * 2, TERRAIN_STEP * 2)
            
            if y + step < self.min_height:
                step = abs(step)
            elif y + step > self.max_height:
                step = -abs(step)

            y += step
            x += self.np_random.uniform(TERRAIN_STEP * 2, TERRAIN_STEP * 4)
            new_poly.append((x, y))
        
        return new_poly
    

    def step(self, action):
        # --- Safely destroy bodies from the PREVIOUS step ---
        for body in self.bodies_to_destroy:
            if body.active:
                self.b2World.DestroyBody(body)
        self.bodies_to_destroy.clear()

        # --- Update the air time counter ---
        self.airborn = len(self.ground_contacts) == 0
        if self.airborn:
            self.air_time_steps += 1
        else:
            self.air_time_steps = 0

        # --- Handle Agent Action ---
        self.step_count += 1
        chassis, wheels, suspensions, human = self.car

        if action == 0:
            if self.airborn: chassis.ApplyTorque(0.0, wake=True)
        elif action == 1:
            self.motorspeed = max(-ACCELERATION, self.motorspeed - 1)
            if self.airborn: chassis.ApplyTorque(TORQUE, wake=True)
        elif action == 2:
            self.motorspeed = min(ACCELERATION, self.motorspeed + 1)
            if self.airborn: chassis.ApplyTorque(-TORQUE, wake=True)
        
        for suspension in suspensions:
            suspension.motorSpeed = self.motorspeed
        
        # --- Step the Physics World ---
        self.b2World.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # --- Infinite Terrain Generation Logic ---
        car_x = self.car[0].position.x
        end_of_terrain_x = self.terrain_poly[-1][0]
        
        # If the car is close to the end, generate a new chunk
        if car_x > end_of_terrain_x - (VIEWPORT_W / SCALE):
            last_x, last_y = self.terrain_poly[-1]
            new_chunk = self._generate_terrain_chunk(last_x, last_y, TERRAIN_LENGTH // 2)
            self.terrain_poly.extend(new_chunk)
            
            # Trim the old part of the terrain to save memory
            self.terrain_poly = self.terrain_poly[- (TERRAIN_LENGTH + TERRAIN_LENGTH // 2):]
            
            # Recreate the physics body and smooth render points
            self.b2World.DestroyBody(self.terrain)
            self.terrain = self.b2World.CreateStaticBody(userData="terrain")
            self.terrain.CreateEdgeChain(self.terrain_poly)


        obs = self._get_observation()

        # --- Calculate Rewards and Queue Coin Destruction ---
        reward = self._calculate_reward()

        # --- Handle Termination and Truncation ---
        truncated = self.step_count > 5000
        #truncated = False

        # Check for being stuck
        PROGRESS_THRESHOLD = 0.1 # meters
        STUCK_LIMIT = 240        # steps (4 seconds)
        LINGER_LIMIT = 300
        current_x = self.car[0].position.x
        if current_x > self.max_x_achieved + PROGRESS_THRESHOLD:
            self.max_x_achieved = current_x
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        if self.steps_since_progress > STUCK_LIMIT:
            self.terminated = True
        if current_x < TERRAIN_STARTPAD and self.step_count > LINGER_LIMIT:
            self.terminated = True

        # Other termination conditions
        if chassis.position.x < TERRAIN_STARTPAD / 2 and self.step_count > 100:
            self.terminated = True
        
        if self.terminated:
            reward = PENALTY_COLLISION

        self.current_score += reward

        if self.render_mode == "human": self.render()
        return obs, reward, self.terminated, truncated, {}

    def _get_observation(self):
        chassis, wheels, suspensions, human = self.car
        pos, vel = chassis.position, chassis.linearVelocity
        
        state = [
            chassis.angle, chassis.angularVelocity, vel.x, vel.y,
            suspensions[0].translation, suspensions[1].translation,
            int(self.airborn)
        ]
        
        # LIDAR Sensor Data
        lidar_readings = []
        for angle in np.linspace(-math.pi / 2, math.pi / 2, 10):
            p1 = pos
            p2 = (pos.x + math.cos(chassis.angle + angle) * 50,
                  pos.y + math.sin(chassis.angle + angle) * 50)
            callback = RayCastCallback()
            self.b2World.RayCast(callback, p1, p2)
            lidar_readings.append(callback.fraction)
        state.extend(lidar_readings)

        # Nearest Coin Data
        if self.enable_coins:
            nearest_coin_pos = None
            if self.coins:
                min_dist_sq = float('inf')
                for coin in self.coins:
                    dist_sq = (pos - coin.position).lengthSquared
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        nearest_coin_pos = coin.position
            if nearest_coin_pos:
                vec_to_coin = nearest_coin_pos - pos
                car_angle = -chassis.angle
                cos_a, sin_a = math.cos(car_angle), math.sin(car_angle)
                local_x = vec_to_coin.x * cos_a - vec_to_coin.y * sin_a
                local_y = vec_to_coin.x * sin_a + vec_to_coin.y * cos_a
                state.extend([local_x, local_y])
            else:
                state.extend([0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def _draw_car_and_driver(self, scroll):
        """Helper method to handle all car and driver rendering."""
        chassis, wheels, _, human = self.car
        
        # Draw Chassis and Wheel Wells
        chassis_vertices = [(chassis.transform * v) * SCALE for v in chassis.fixtures[0].shape.vertices]
        chassis_vertices_screen = [(v[0] - scroll, VIEWPORT_H - v[1]) for v in chassis_vertices]
        pygame.draw.polygon(self.screen, (220, 0, 0), chassis_vertices_screen)
    
        # Draw Wheels
        for wheel in wheels:
            pos = (wheel.position * SCALE)
            pos_screen = (pos[0] - scroll, VIEWPORT_H - pos[1])
            pygame.draw.circle(self.screen, (40, 40, 40), pos_screen, wheel.fixtures[0].shape.radius * SCALE)
            pygame.draw.circle(self.screen, (150, 150, 150), pos_screen, wheel.fixtures[0].shape.radius * SCALE * 0.6, 4)

        # Draw Driver's Body and Head
        human_body_fixture = human.fixtures[0]
        human_head_fixture = human.fixtures[1]

        body_vertices = [(human.transform * v) * SCALE for v in human_body_fixture.shape.vertices]
        body_vertices_screen = [(v[0] - scroll, VIEWPORT_H - v[1]) for v in body_vertices]
        pygame.draw.polygon(self.screen, (50, 50, 200), body_vertices_screen)

        head_world_pos = human.GetWorldPoint(localPoint=human_head_fixture.shape.pos)
        head_screen_pos = (head_world_pos.x * SCALE - scroll, VIEWPORT_H - head_world_pos.y * SCALE)
        head_radius_screen = human_head_fixture.shape.radius * SCALE
        pygame.draw.circle(self.screen, (255, 255, 255), head_screen_pos, head_radius_screen)

        # Draw Steering Wheel and Arm
        steering_wheel_pos_world = chassis.GetWorldPoint(localPoint=(-0.3, 0.25))
        shoulder_pos_world = human.GetWorldPoint(localPoint=(0.1, 0.2))
        steering_wheel_pos_screen = (steering_wheel_pos_world.x * SCALE - scroll, VIEWPORT_H - steering_wheel_pos_world.y * SCALE)
        shoulder_pos_screen = (shoulder_pos_world.x * SCALE - scroll, VIEWPORT_H - shoulder_pos_world.y * SCALE)
        pygame.draw.line(self.screen, (50, 50, 200), shoulder_pos_screen, steering_wheel_pos_screen, 7)
        pygame.draw.circle(self.screen, (40, 40, 40), steering_wheel_pos_screen, 0.15 * SCALE, 4)

        # Draw Helmet Visor
        head_rect = pygame.Rect(head_screen_pos[0] - head_radius_screen, head_screen_pos[1] - head_radius_screen, head_radius_screen * 2, head_radius_screen * 2)
        visor_angle = human.angle + math.pi / 2
        pygame.draw.arc(self.screen, (0, 0, 0), head_rect, visor_angle - 0.7, visor_angle + 0.9, 4)


    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            pygame.display.set_caption("Hill Climb RL")
            self.font = pygame.font.Font(None, 40)
        if self.clock is None: self.clock = pygame.time.Clock()

        if self.b2World is None: return

        self.screen.fill(BACKGROUND_COLOR)
        
        scroll = self.car[0].position.x * SCALE - VIEWPORT_W / 4
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.terminated = True
                elif event.key == pygame.K_q:
                    self.close()

        # --- Draw Terrain ---
        if self.smooth_terrain_poly:
            smooth_vertices = [(v[0] * SCALE - scroll, VIEWPORT_H - v[1] * SCALE) for v in self.smooth_terrain_poly]
            if len(smooth_vertices) > 1:
                polygon_points = list(smooth_vertices)
                polygon_points.append((smooth_vertices[-1][0], VIEWPORT_H))
                polygon_points.append((smooth_vertices[0][0], VIEWPORT_H))
                pygame.draw.polygon(self.screen, SOIL_COLOR, polygon_points)
                pygame.draw.lines(self.screen, GRASS_COLOR, False, smooth_vertices, 5)

        # --- Draw the Wall ---
        if self.wall:
            # The wall is a polygon, so we find its vertices and draw it
            for fixture in self.wall.fixtures:
                shape = fixture.shape
                vertices = [(self.wall.transform * v) * SCALE for v in shape.vertices]
                vertices_screen = [(v[0] - scroll, VIEWPORT_H - v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, SOIL_COLOR, vertices_screen)

        # --- Draw Coins ---
        for coin in self.coins:
            pos = (coin.position * SCALE)
            pos = (pos[0] - scroll, VIEWPORT_H - pos[1])
            pygame.draw.circle(self.screen, COIN_COLOR, pos, 0.3 * SCALE)

        # --- Draw Car and Driver ---
        self._draw_car_and_driver(scroll)

        # --- Render Score and Coin Text ---
        if not self.terminated:
            score_text = f"Score: {int(self.current_score)}"
            coin_text = f"Coins: {self.coins_collected}"
            
            score_surface = self.font.render(score_text, True, (0, 0, 0))
            coin_surface = self.font.render(coin_text, True, (128,128,0))
            
            self.screen.blit(score_surface, (15, 15))
            self.screen.blit(coin_surface, (15, 55))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            
