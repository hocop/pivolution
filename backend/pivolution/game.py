import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
from numba import njit
import multiprocessing
from perlin_noise import PerlinNoise

from .creature import Creature


class Game:
    def __init__(self, map_h=720//4, map_w=1280//4, render_h=720, render_w=1280, min_seconds_per_step=0, min_seconds_per_render=0, seed=43, map_seed=42):
        self.map_h = map_h
        self.map_w = map_w
        self.render_h = render_h
        self.render_w = render_w
        self.min_seconds_per_step = min_seconds_per_step
        self.min_seconds_per_render = min_seconds_per_render

        self.info = {}
        self.rendering = False
        np.random.seed(seed)

        # Elevation map
        self.elevation_map = self.generate_elevation(map_seed)
        self.water_depth_map = np.clip(-self.elevation_map, 0, None)

        # Some maps
        self.creature_id_map = -np.ones([map_h, map_w], dtype='int')
        self.meat_map = np.zeros([map_h, map_w], dtype='float')

        # Creatures
        self.creatures = []
        self.dead_creatures_ids = []
        self.num_creatures = 0

        # Time stamps
        self.previous_render_time = time.time()
        self.previous_step_time = time.time()
        

    def render(self, camera_x=None, camera_y=None, scale=4):
        start = time.time()
        self.rendering = True
        # Get image
        if camera_x is None:
            camera_x = self.map_w // 2
        if camera_y is None:
            camera_y = self.map_h // 2
        crop = [camera_y - self.render_h // 2 // scale, camera_y + self.render_h // 2 // scale, camera_x - self.render_w // 2 // scale, camera_x + self.render_w // 2 // scale]
        elevation = pad_and_crop(self.elevation_map, *crop)
        creature_id_map = pad_and_crop(self.creature_id_map, *crop, padding=-1)
        meat_map = pad_and_crop(self.meat_map, *crop)

        # Draw elevation
        elev_min, elev_max = -2, 2
        image = (elevation.clip(elev_min, elev_max) - elev_min) / (elev_max - elev_min)
        image = (image * 255).astype('uint8')
        image = np.repeat(image[:, :, None], 3, 2)
        # Draw water
        water_color = (-elevation * 127).clip(0, 75)
        water_image = np.zeros_like(image)
        water_image[:, :, 0] = 75 - water_color
        water_image[:, :, 1] = 75 - water_color
        water_image[:, :, 2] = 127
        water_mask = elevation < 0
        image[water_mask] = water_image[water_mask]
        # Draw meat
        meat_image = np.zeros_like(image)
        meat_mask = np.clip(meat_map, 0, 1)
        meat_image[:, :, 0] = 255 * meat_mask
        meat_image[:, :, 1] = 0
        meat_image[:, :, 2] = 255 * meat_mask
        image = (image * (1 - meat_mask[:, :, None]) + meat_image * meat_mask[:, :, None]).astype('uint8')
        # Draw creatures
        if scale < 5:
            for idx in np.unique(creature_id_map):
                if idx < 0 or self.creatures[idx] is None:
                    continue
                creature, pos_x, pos_y, angle = self.creatures[idx]
                pos_x, pos_y = self.global_to_camera(pos_x, pos_y, camera_x, camera_y, scale)
                if pos_x >= 0 and pos_y >= 0 and pos_x < image.shape[1] and pos_y < image.shape[0]:
                    image[pos_y, pos_x, :] = creature.color
        # Resize image
        image = np.repeat(image, scale, 0)
        image = np.repeat(image, scale, 1)

        elapsed = time.time() - self.previous_render_time
        if elapsed < self.min_seconds_per_render:
            time.sleep(self.min_seconds_per_render - elapsed)
        self.info['render_freq'] = int(1. / (time.time() - self.previous_render_time))
        self.info['ns_render'] = int((time.time() - start) * 1e6 / (self.num_creatures + 1))
        print(self.info)
        self.previous_render_time = time.time()
        self.rendering = False
        return image

    def global_to_camera(self, pos_x, pos_y, camera_x, camera_y, scale):
        left = camera_x - self.render_w // 2 // scale
        top = camera_y - self.render_h // 2 // scale
        return pos_x - left, pos_y - top

    @staticmethod
    def compute_action(creature):
        if creature is None:
            return
        creature[0].compute_action()
        return creature

    def step(self):
        start = time.time()

        # Creature params map
        predatory_map = np.zeros([self.map_h, self.map_w])
        energy_map = np.zeros([self.map_h, self.map_w])
        health_map = np.zeros([self.map_h, self.map_w])
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            predatory_map[pos_y, pos_x] = creature.predatory
            energy_map[pos_y, pos_x] = creature.energy
            health_map[pos_y, pos_x] = creature.health

        # Get features
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            # Get features
            water_depth = self.water_depth_map[pos_y, pos_x]
            meat_in_this_cell = self.meat_map[pos_y, pos_x]
            # Get creature in front
            creature_in_front = None
            front_x, front_y = self.get_cell_in_front(pos_x, pos_y, angle)
            if front_x >= 0 and front_x < self.map_w and front_y >= 0 and front_y < self.map_h:
                front_idx = self.creature_id_map[front_y, front_x]
                if front_idx >= 0:
                    creature_in_front = self.creatures[front_idx][0]
            # Get local features
            local_feats = np.concatenate([
                pad_and_crop(feat_map, pos_y - 2, pos_y + 3, pos_x - 2, pos_x + 3)[None]
                for feat_map in [predatory_map, energy_map, health_map, self.water_depth_map, self.meat_map]
            ], 0)
            local_feats = self.rotate_feats(local_feats, angle)
            # Set features
            creature.features = water_depth, meat_in_this_cell, creature_in_front, local_feats
        end_feats = time.time()

        # Compute actions
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature = self.creatures[idx][0]
            creature.compute_action()
        # with multiprocessing.Pool() as pool:
        #     self.creatures = pool.map(self.compute_action, self.creatures)
        end_actions = time.time()

        # Remove deadge
        num_creatures = 0
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            if creature.health <= 0:
                if self.creature_id_map[pos_y, pos_x] == idx:
                    self.creature_id_map[pos_y, pos_x] = -1
                meat = max(0.05, creature.energy) * (1 - creature.predatory)
                self.meat_map[pos_y, pos_x] = min(1.0, self.meat_map[pos_y, pos_x] + meat)
                self.dead_creatures_ids.append(idx)
                self.creatures[idx] = None
            else:
                num_creatures = num_creatures + 1
        self.num_creatures = num_creatures
        end_deadge = time.time()

        # Apply actions
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            action = creature.action
            # Eat meat
            if creature.predatory > 0.5:
                self.meat_map[pos_y, pos_x] = 0
            # Reproduce
            if action == 'reproduce':
                spawn_x, spawn_y = self.get_cell_in_front(pos_x, pos_y, angle, direction=-1)
                spawn_angle = (angle + 1) % 4
                if spawn_x >= 0 and spawn_x < self.map_w and spawn_y >= 0 and spawn_y < self.map_h:
                    behind_idx = self.creature_id_map[spawn_y, spawn_x]
                    if behind_idx < 0 or self.creatures[behind_idx] is None:
                        # Spawn new creature
                        self.spawn(spawn_x, spawn_y, spawn_angle, creature.get_offspring())
            # Go forward or backward
            new_pos_x, new_pos_y, new_angle = pos_x, pos_y, angle
            if 'go' in action:
                direction = 1 if action == 'go forward' else -1
                new_pos_x, new_pos_y = self.get_cell_in_front(pos_x, pos_y, angle, direction)
                # Boundaries
                if new_pos_x < 0:
                    new_pos_x = 0
                if new_pos_y < 0:
                    new_pos_y = 0
                if new_pos_x >= self.map_w:
                    new_pos_x = self.map_w - 1
                if new_pos_y >= self.map_h:
                    new_pos_y = self.map_h - 1
                # Check collision
                go_id = self.creature_id_map[new_pos_y, new_pos_x]
                if go_id >= 0 and self.creatures[go_id] is not None:
                    new_pos_y, new_pos_x = pos_y, pos_x
            # Rotate
            elif 'turn' in action:
                direction = 1 if action == 'turn right' else -1
                new_angle = (angle + direction) % 4
            # Update data
            self.creatures[idx][1] = new_pos_x
            self.creatures[idx][2] = new_pos_y
            self.creatures[idx][3] = new_angle
            # Update indeces
            self.creature_id_map[pos_y, pos_x] = -1
            self.creature_id_map[new_pos_y, new_pos_x] = idx
        end_apply = time.time()

        elapsed = time.time() - self.previous_step_time
        if elapsed < self.min_seconds_per_step:
            time.sleep(self.min_seconds_per_step - elapsed)
        self.info['sim_freq'] = int(1. / (time.time() - self.previous_step_time))
        self.info['ns_sim'] = int((time.time() - start) * 1e6 / (self.num_creatures + 1))
        self.info['ns_feats'] = int((end_feats - start) * 1e6 / (self.num_creatures + 1))
        self.info['ns_actions'] = int((end_actions - end_feats) * 1e6 / (self.num_creatures + 1))
        self.info['ns_deadge'] = int((end_deadge - end_actions) * 1e6 / (self.num_creatures + 1))
        self.info['ns_apply'] = int((end_apply - end_deadge) * 1e6 / (self.num_creatures + 1))
        self.info['num_creatures'] = num_creatures
        self.previous_step_time = time.time()

    def rotate_feats(self, feats, angle):
        if angle == 1:
            feats = feats.transpose(0, 2, 1)[:, ::-1]
        if angle == 2:
            feats = feats[:, ::-1, ::-1]
        if angle == 3:
            feats = feats.transpose(0, 2, 1)[:, :, ::-1]
        return feats

    def spawn(self, pos_x=None, pos_y=None, angle=None, creature=None):
        # Generate random
        if pos_x is None:
            pos_x = np.random.randint(self.map_w)
        if pos_y is None:
            pos_y = np.random.randint(self.map_h)
        if angle is None:
            angle = np.random.randint(4)
        if creature is None:
            creature = Creature()

        # Find index with None value
        new_idx = None
        for i in range(len(self.dead_creatures_ids)):
            idx = self.dead_creatures_ids[i]
            if self.creatures[idx] == None:
                new_idx = idx
                self.dead_creatures_ids.pop(i)
                break
            else:
                print('Dead error')

        # Spawn
        if new_idx is None:
            self.creatures.append([creature, pos_x, pos_y, angle])
            self.creature_id_map[pos_y, pos_x] = len(self.creatures) - 1
        else:
            self.creatures[new_idx] = [creature, pos_x, pos_y, angle]
            self.creature_id_map[pos_y, pos_x] = new_idx

    def generate_elevation(self, seed):
        print('Generating map...')
        scale = max(self.map_h, self.map_w)
        noise1 = PerlinNoise(octaves=scale//80, seed=seed)
        noise2 = PerlinNoise(octaves=scale//40, seed=seed)
        noise3 = PerlinNoise(octaves=scale//20, seed=seed)
        noise4 = PerlinNoise(octaves=scale//10, seed=seed)
        elevation = []
        for i in range(self.map_h):
            row = []
            for j in range(self.map_w):
                noise_val = noise1([i / scale, j / scale])
                noise_val += 0.5 * noise2([i / scale, j / scale])
                noise_val += 0.25 * noise3([i / scale, j / scale])
                noise_val += 0.125 * noise4([i / scale, j / scale])

                row.append(noise_val)
            elevation.append(row)
        elevation = np.array(elevation)
        print('elevation', elevation.shape)
        return elevation


    def get_cell_in_front(self, pos_x, pos_y, angle, direction=1):
        new_pos_x, new_pos_y = pos_x, pos_y
        if angle == 0:
            new_pos_x = pos_x + direction
        elif angle == 1:
            new_pos_y = pos_y + direction
        elif angle == 2:
            new_pos_x = pos_x - direction
        elif angle == 3:
            new_pos_y = pos_y - direction
        return new_pos_x, new_pos_y


def pad_and_crop(img, hs, he, ws, we, padding=0):
    # Pad
    if hs < 0:
        img = np.concatenate([np.full([-hs, img.shape[1]], padding, dtype=img.dtype), img], 0)
        he = he - hs
        hs = 0
    if ws < 0:
        img = np.concatenate([np.full([img.shape[0], -ws], padding, dtype=img.dtype), img], 1)
        we = we - ws
        ws = 0
    if he > img.shape[0]:
        img = np.concatenate([img, np.full([he - img.shape[0], img.shape[1]], padding, dtype=img.dtype)], 0)
    if we > img.shape[1]:
        img = np.concatenate([img, np.full([img.shape[0], we - img.shape[1]], padding, dtype=img.dtype)], 1)
    # Crop
    return img[hs: he, ws: we]