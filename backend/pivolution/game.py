import pickle
import numpy as np
import time
from numba import njit
from perlin_noise import PerlinNoise
import scipy

from .creatures import CreatureRandom, CreatureLinear, CreatureNeural, CreatureRecurrent, CreatureGendered
from .creatures.basic import FEAT_WINDOW


PORTAL_SIZE = 0.3
PORTAL_COLOR = [0, 255, 200]
PORTAL_DISABLED_COLOR = [100, 100, 100]

WORLD_MARGIN = 1


class Game:
    def __init__(self, map_h=720//4, map_w=1280//4, default_scale=4, seed=43, map_seed=42, subworld_id=None, elevation=None, save_period=-1):
        self.map_h = map_h
        self.map_w = map_w
        self.default_scale = default_scale
        self.render_h = map_h * default_scale
        self.render_w = map_w * default_scale
        self.subworld_id = subworld_id
        self.seed = seed
        self.save_period = save_period

        self.info = {}
        self.rendering = False
        self.steps_count = 0
        self.portals = {}
        self.portal_out_enabled = False
        np.random.seed(seed)

        # Elevation map
        self.elevation_map, self.walls_map = self.generate_elevation_and_walls(map_seed, elevation)
        self.water_depth_map = np.clip(-self.elevation_map, 0, None)

        # Some maps
        self.creature_id_map = -np.ones([map_h, map_w], dtype='int')
        self.meat_map = np.zeros([map_h, map_w], dtype='float')

        # Creatures
        self.creatures = []
        self.removed_creatures_ids = []
        self.num_creatures = 0
        self.creatures_queue = []

        # Time stamps
        self.previous_render_time = time.time()
        self.previous_step_time = time.time()

        # Pre-render
        # Draw elevation
        elev_min, elev_max = -2, 2
        image = (self.elevation_map.clip(elev_min, elev_max) - elev_min) / (elev_max - elev_min)
        image = image * 255
        image = np.repeat(image[:, :, None], 3, 2)
        # Draw walls
        image = image * (1 - self.walls_map[:, :, None] * 0.5)
        # Draw water
        water_alpha = (-self.elevation_map + 0.4).clip(0, 1) * (self.elevation_map < 0)
        water_alpha = water_alpha[:, :, None]
        water_image = np.zeros_like(image)
        water_image[:, :, 0] = 0
        water_image[:, :, 1] = 0
        water_image[:, :, 2] = 127
        image = image * (1 - water_alpha) + water_image * water_alpha
        # Save image
        self.landscape_image = image.astype('uint8')


    def render(self, camera_x=None, camera_y=None, scale=None):
        start = time.time()
        self.rendering = True
        # Get image
        scale = scale or self.default_scale
        if camera_x is None:
            camera_x = self.map_w // 2
        if camera_y is None:
            camera_y = self.map_h // 2
        crop = [camera_y - self.render_h // 2 // scale, camera_y + self.render_h // 2 // scale, camera_x - self.render_w // 2 // scale, camera_x + self.render_w // 2 // scale]
        image = crop_and_pad(self.landscape_image, *crop).copy()
        creature_id_map = crop_and_pad(self.creature_id_map, *crop, cval=-1)
        meat_map = crop_and_pad(self.meat_map, *crop)

        # Draw meat
        meat_image = np.zeros_like(image)
        meat_mask = np.clip(meat_map, 0, 1)
        meat_image[:, :, 0] = 200 * meat_mask
        meat_image[:, :, 2] = 255 * meat_mask
        image = (image * (1 - meat_mask[:, :, None]) + meat_image * meat_mask[:, :, None]).astype('uint8')
        # Get creature coords and colors
        arr_pos_x, arr_pos_y, arr_angle, arr_color, arr_face_color, arr_mid_color = [], [], [], [], [], []
        for idx in np.unique(creature_id_map):
            if idx < 0 or self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            pos_x, pos_y = self.global_to_camera(pos_x, pos_y, camera_x, camera_y, scale)
            arr_pos_x.append(pos_x)
            arr_pos_y.append(pos_y)
            arr_angle.append(angle)
            arr_color.append(creature.color)
            arr_face_color.append(creature.face_color)
            arr_mid_color.append(creature.middle_color)
        arr_pos_x, arr_pos_y, arr_angle, arr_color, arr_face_color = map(np.array, [arr_pos_x, arr_pos_y, arr_angle, arr_color, arr_face_color])
        # Draw creatures
        if len(arr_color) > 0:
            image[arr_pos_y, arr_pos_x] = arr_color
        # Resize image
        image = np.repeat(image, scale, 0)
        image = np.repeat(image, scale, 1)
        # Draw creature faces
        if scale >= 2 and len(arr_color) > 0:
            mask = arr_angle == 0
            pos_x, pos_y, color = arr_pos_x[mask], arr_pos_y[mask], arr_face_color[mask]
            for i in range(scale):
                image[pos_y * scale + i, pos_x * scale + scale - 1] = color
            mask = arr_angle == 1
            pos_x, pos_y, color = arr_pos_x[mask], arr_pos_y[mask], arr_face_color[mask]
            for j in range(scale):
                image[pos_y * scale + scale - 1, pos_x * scale + j] = color
            mask = arr_angle == 2
            pos_x, pos_y, color = arr_pos_x[mask], arr_pos_y[mask], arr_face_color[mask]
            for i in range(scale):
                image[pos_y * scale + i, pos_x * scale] = color
            mask = arr_angle == 3
            pos_x, pos_y, color = arr_pos_x[mask], arr_pos_y[mask], arr_face_color[mask]
            for j in range(scale):
                image[pos_y * scale, pos_x * scale + j] = color
        # Draw middle points
        has_mid = np.array([mc is not None for mc in arr_mid_color])
        arr_mid_color = np.array([mc for mc in arr_mid_color if mc is not None])
        if len(arr_mid_color) > 0:
            pos_x, pos_y = arr_pos_x[has_mid], arr_pos_y[has_mid]
            a, b = scale // 2 - 1, scale // 2 + 1
            if scale >= 2 and np.sum(has_mid) > 0:
                for i in range(a, b):
                    for j in range(a, b):
                        image[pos_y * scale + i, pos_x * scale + j] = arr_mid_color
        # Draw portals
        w = image.shape[1]
        margin_up = np.zeros([WORLD_MARGIN, w, 3], dtype='uint8')
        margin_down = margin_up.copy()
        portal_color = PORTAL_COLOR if self.portal_out_enabled else PORTAL_DISABLED_COLOR
        if 'up' in self.portals:
            margin_up[:, int(w - w * PORTAL_SIZE) // 2: int(w + w * PORTAL_SIZE) // 2] = portal_color
        if 'down' in self.portals:
            margin_down[:, int(w - w * PORTAL_SIZE) // 2: int(w + w * PORTAL_SIZE) // 2] = portal_color
        image = np.concatenate([margin_up, image, margin_down], 0)
        h = image.shape[0]
        margin_left = np.zeros([h, WORLD_MARGIN, 3], dtype='uint8')
        margin_right = margin_left.copy()
        if 'left' in self.portals:
            margin_left[int(h - h * PORTAL_SIZE) // 2: int(h + h * PORTAL_SIZE) // 2, :] = portal_color
        if 'right' in self.portals:
            margin_right[int(h - h * PORTAL_SIZE) // 2: int(h + h * PORTAL_SIZE) // 2, :] = portal_color
        image = np.concatenate([margin_left, image, margin_right], 1)

        end = time.time()
        self.info['μs_render'] = int((end - start) * 1e6 / (self.num_creatures + 1))
        self.info['render_total_ms'] = int((end - start) * 1000)
        if 'sim_total_ms' in self.info:
            ms_with_wait = int((end - self.previous_render_time) * 1000)
            self.info['efficiency'] = (self.info['render_total_ms'] + self.info['sim_total_ms']) / ms_with_wait
        # if self.subworld_id is None:
        print(self.subworld_id, self.info)
        self.previous_render_time = time.time()
        self.rendering = False
        return image

    def global_to_camera(self, pos_x, pos_y, camera_x, camera_y, scale):
        left = camera_x - self.render_w // 2 // scale
        top = camera_y - self.render_h // 2 // scale
        return pos_x - left, pos_y - top

    def step(self):
        start = time.time()

        # Unstuck some creatures stuck in portals
        if len(self.creatures_queue) > 0:
            creature, pos_x, pos_y, angle = self.creatures_queue[0]
            self.creatures_queue = self.creatures_queue[1:]
            self.spawn(creature, pos_x, pos_y, angle)

        # Recieve creatures from portals
        for dir in self.portals:
            q_in = self.portals[dir]['in']
            if q_in is not None and not q_in.empty():
                signal = q_in.get()
                if signal == 'stop':
                    self.portals[dir]['in'] = None
                else:
                    creature, pos_x, pos_y, angle = signal
                    # Periodic conditions
                    pos_x = pos_x % self.map_w
                    pos_y = pos_y % self.map_h
                    self.spawn(creature, pos_x, pos_y, angle)
                    # print(self.subworld_id, 'RECIEVED', pos_x, pos_y, self.creature_id_map[pos_y, pos_x])

        # Creature params map
        predatory_map = np.full([self.map_h, self.map_w], -1)
        energy_map = np.full([self.map_h, self.map_w], -1)
        health_map = np.full([self.map_h, self.map_w], -1)
        direction_map = np.full([self.map_h, self.map_w], 0)
        gender_map = np.full([self.map_h, self.map_w], -1)
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            predatory_map[pos_y, pos_x] = creature.predatory
            energy_map[pos_y, pos_x] = creature.energy
            health_map[pos_y, pos_x] = creature.health
            if isinstance(creature, CreatureGendered):
                gender_map[pos_y, pos_x] = 0 if creature.gender == 'boy' else 1
            front_x, front_y = self.get_cell_in_front(pos_x, pos_y, angle)
            if self.is_valid_coords(front_x, front_y):
                direction_map[front_y, front_x] = 1

        # Get features
        win_left = FEAT_WINDOW // 2
        win_right = FEAT_WINDOW // 2 + 1
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            # Get features
            water_depth = self.water_depth_map[pos_y, pos_x]
            air_level = max(self.elevation_map[pos_y, pos_x], 0)
            meat_in_this_cell = self.meat_map[pos_y, pos_x]
            # Get creature in front
            creature_in_front = None
            front_x, front_y = self.get_cell_in_front(pos_x, pos_y, angle)
            if self.is_valid_coords(front_x, front_y):
                front_idx = self.creature_id_map[front_y, front_x]
                if front_idx >= 0:
                    creature_in_front = self.creatures[front_idx][0]
            # Get local features
            local_feats = np.concatenate([
                crop_and_pad(feat_map, pos_y - win_left, pos_y + win_right, pos_x - win_left, pos_x + win_right)[None]
                for feat_map in [predatory_map, energy_map, health_map, direction_map, gender_map, self.water_depth_map, self.walls_map, self.meat_map]
            ], 0)
            local_feats = self.rotate_feats(local_feats, angle)
            # Set features
            creature.features = water_depth, air_level, meat_in_this_cell, creature_in_front, local_feats
        end_feats = time.time()

        # Compute actions
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature = self.creatures[idx][0]
            creature.compute_action()
        end_actions = time.time()

        # Remove deadge
        num_creatures = 0
        for idx in range(len(self.creatures)):
            if self.creatures[idx] is None:
                continue
            creature, pos_x, pos_y, angle = self.creatures[idx]
            assert self.creature_id_map[pos_y, pos_x] == idx, (self.creature_id_map[pos_y, pos_x], idx)
            if creature.health <= 0:
                self.remove_creature(pos_x, pos_y, leave_meat=True)
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
            if creature.gain_from_meat > 0:
                self.meat_map[pos_y, pos_x] = 0
            # Reproduce
            if action == 'reproduce':
                spawn_x, spawn_y = self.get_cell_in_front(pos_x, pos_y, angle, direction=-1)
                spawn_angle = (angle + 1) % 4
                if self.is_valid_coords(spawn_x, spawn_y):
                    behind_idx = self.creature_id_map[spawn_y, spawn_x]
                    if behind_idx < 0 or self.creatures[behind_idx] is None:
                        # Spawn new creature
                        offspring = creature.reproduce()
                        if offspring is not None:
                            self.spawn(offspring, spawn_x, spawn_y, spawn_angle)
            # Go forward or backward
            new_pos_x, new_pos_y, new_angle = pos_x, pos_y, angle
            if 'go' in action:
                direction = 1 if action == 'go forward' else -1
                new_pos_x, new_pos_y = self.get_cell_in_front(pos_x, pos_y, angle, direction)
                # Check output portals
                if self.portal_out_enabled:
                    teleported = False
                    for dir in self.portals:
                        if self.check_in_portal(new_pos_x, new_pos_y, dir):
                            self.portals[dir]['out'].put((creature, new_pos_x, new_pos_y, angle))
                            self.remove_creature(pos_x, pos_y, leave_meat=False)
                            # print(self.subworld_id, 'SENT CREATURE', pos_x, pos_y)
                            teleported = True
                            break
                    if teleported:
                        self.num_creatures -= 1
                        continue
                # Boundaries
                if not(self.is_valid_coords(new_pos_x, new_pos_y)):
                    new_pos_x, new_pos_y = pos_x, pos_y
                # Check collision
                go_id = self.creature_id_map[new_pos_y, new_pos_x]
                if go_id >= 0:
                    assert self.creatures[go_id] is not None
                    new_pos_y, new_pos_x = pos_y, pos_x
            # Rotate
            if 'turn' in action:
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

        end = time.time()
        self.info['sim_total_ms'] = int((end - start) * 1000)
        self.info['μs_sim'] = int((end - start) * 1e6 / (self.num_creatures + 1))
        self.info['μs_feats'] = int((end_feats - start) * 1e6 / (self.num_creatures + 1))
        self.info['μs_actions'] = int((end_actions - end_feats) * 1e6 / (self.num_creatures + 1))
        self.info['μs_deadge'] = int((end_deadge - end_actions) * 1e6 / (self.num_creatures + 1))
        self.info['μs_apply'] = int((end_apply - end_deadge) * 1e6 / (self.num_creatures + 1))
        self.info['num_creatures'] = num_creatures
        self.previous_step_time = time.time()

        if self.save_period > 0 and self.steps_count % self.save_period == 0:
            fname = f'world_{self.steps_count:08n}.pickle'
            with open(fname, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Saved as', fname)
        self.steps_count += 1

    def rotate_feats(self, feats, angle):
        if angle == 1:
            feats = feats.transpose(0, 2, 1)[:, ::-1]
        if angle == 2:
            feats = feats[:, ::-1, ::-1]
        if angle == 3:
            feats = feats.transpose(0, 2, 1)[:, :, ::-1]
        return feats

    def spawn(self, creature=None, pos_x=None, pos_y=None, angle=None):
        # Find index with None value
        new_idx = None
        for i in range(len(self.removed_creatures_ids)):
            idx = self.removed_creatures_ids[i]
            if self.creatures[idx] == None:
                new_idx = idx
                self.removed_creatures_ids.pop(i)
                break
            else:
                raise BaseException('Dead error')

        # Seed
        if len(self.creatures) == 0:
            np.random.seed(self.seed)

        # Generate random
        if pos_x is None:
            assert pos_y is None
        if pos_y is None:
            assert pos_x is None
            pos_x = np.random.randint(self.map_w)
            pos_y = np.random.randint(self.map_h)
            while(not (self.is_valid_coords(pos_x, pos_y) and self.creature_id_map[pos_y, pos_x] == -1)):
                pos_x = np.random.randint(self.map_w)
                pos_y = np.random.randint(self.map_h)
        else:
            assert self.is_valid_coords(pos_x, pos_y)
        if angle is None:
            angle = np.random.randint(4)
        if creature is None:
            creature = CreatureGendered()

        # Add to queue if cannot spawn right now
        if self.creature_id_map[pos_y, pos_x] >= 0:
            self.creatures_queue.append([creature, pos_x, pos_y, angle])
            return

        # Spawn
        if new_idx is None:
            self.creatures.append([creature, pos_x, pos_y, angle])
            self.creature_id_map[pos_y, pos_x] = len(self.creatures) - 1
        else:
            self.creatures[new_idx] = [creature, pos_x, pos_y, angle]
            self.creature_id_map[pos_y, pos_x] = new_idx

    def remove_creature(self, pos_x, pos_y, leave_meat):
        idx = self.creature_id_map[pos_y, pos_x]
        # Add meat to map
        if leave_meat:
            meat = 0.25 * (1 - self.creatures[idx][0].predatory)
            self.meat_map[pos_y, pos_x] = min(4.0, self.meat_map[pos_y, pos_x] + meat)
        # Remove
        self.removed_creatures_ids.append(idx)
        self.creature_id_map[pos_y, pos_x] = -1
        self.creatures[idx] = None

    def add_portal(self, where, queue_in, queue_out):
        self.portals[where] = {'in': queue_in, 'out': queue_out}
        self.portal_out_enabled = True
    
    def check_in_portal(self, pos_x, pos_y, dir):
        if pos_y < 0 and dir == 'up':
            low = self.map_w * 0.5 * (1 - PORTAL_SIZE)
            high = self.map_w * 0.5 * (1 + PORTAL_SIZE)
            if pos_x > low and pos_x < high:
                return True
        if pos_y >= self.map_h and dir == 'down':
            low = self.map_w * 0.5 * (1 - PORTAL_SIZE)
            high = self.map_w * 0.5 * (1 + PORTAL_SIZE)
            if pos_x > low and pos_x < high:
                return True
        if pos_x < 0 and dir == 'left':
            low = self.map_h * 0.5 * (1 - PORTAL_SIZE)
            high = self.map_h * 0.5 * (1 + PORTAL_SIZE)
            if pos_y > low and pos_y < high:
                return True
        if pos_x >= self.map_w and dir == 'right':
            low = self.map_h * 0.5 * (1 - PORTAL_SIZE)
            high = self.map_h * 0.5 * (1 + PORTAL_SIZE)
            if pos_y > low and pos_y < high:
                return True
        return False

    def destroy_portals(self):
        # Stop sending creatures
        self.portal_out_enabled = False
        # Send stop signal
        for dir in self.portals:
            self.portals[dir]['out'].put('stop')
        # Wait until there is no creatures stuck inside portals
        for dir in self.portals:
            while self.portals[dir]['in'] is not None:
                self.step()
        # Remove portals
        self.portals = {}

    def generate_elevation_and_walls(self, seed, elevation=None):
        if elevation is None:
            elevation = generate_elevation(self.map_h, self.map_w, seed)

        grad_y = scipy.ndimage.sobel(elevation, axis=0)
        grad_x = scipy.ndimage.sobel(elevation, axis=1)
        walls = (grad_x**2 + grad_y**2) > 0.05
        walls = scipy.ndimage.binary_erosion(walls, iterations=2)

        return elevation, walls


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
    
    def is_valid_coords(self, pos_x, pos_y):
        return pos_x >= 0 and pos_x < self.map_w and pos_y >= 0 and pos_y < self.map_h and not self.walls_map[pos_y, pos_x]



def generate_elevation(map_h, map_w, seed):
    print('Generating map...')
    scale = max(map_h, map_w)
    noise1 = PerlinNoise(octaves=scale//80, seed=seed)
    noise2 = PerlinNoise(octaves=scale//40, seed=seed)
    noise3 = PerlinNoise(octaves=scale//20, seed=seed)
    noise4 = PerlinNoise(octaves=scale//10, seed=seed)
    elevation = []
    for i in range(map_h):
        row = []
        for j in range(map_w):
            noise_val = noise1([i / scale, j / scale])
            noise_val += 0.5 * noise2([i / scale, j / scale])
            noise_val += 0.25 * noise3([i / scale, j / scale])
            noise_val += 0.125 * noise4([i / scale, j / scale])

            row.append(noise_val)
        elevation.append(row)
    elevation = np.array(elevation)
    print('elevation', elevation.shape)
    return elevation



def pad_and_crop(img, hs, he, ws, we, cval=0):
    # Pad
    if hs < 0:
        img = np.concatenate([np.full([-hs, img.shape[1]], cval, dtype=img.dtype), img], 0)
        he = he - hs
        hs = 0
    if ws < 0:
        img = np.concatenate([np.full([img.shape[0], -ws], cval, dtype=img.dtype), img], 1)
        we = we - ws
        ws = 0
    if he > img.shape[0]:
        img = np.concatenate([img, np.full([he - img.shape[0], img.shape[1]], cval, dtype=img.dtype)], 0)
    if we > img.shape[1]:
        img = np.concatenate([img, np.full([img.shape[0], we - img.shape[1]], cval, dtype=img.dtype)], 1)
    # Crop
    return img[hs: he, ws: we]


# @njit
def crop_and_pad(img, hs, he, ws, we, cval=0):
    # Crop
    hs_real = hs
    if hs < 0:
        hs_real = 0
    he_real = he
    if he > img.shape[0]:
        he_real = img.shape[0]
    ws_real = ws
    if ws < 0:
        ws_real = 0
    we_real = we
    if we > img.shape[1]:
        we_real = img.shape[1]
    img = img[hs_real: he_real, ws_real: we_real]
    # Pad
    if hs < 0:
        pad = np.full((-hs, *img.shape[1:]), cval, dtype=img.dtype)
        img = np.concatenate((pad, img), 0)
    if ws < 0:
        pad = np.full((img.shape[0], -ws, *img.shape[2:]), cval, dtype=img.dtype)
        img = np.concatenate((pad, img), 1)
    if he > he_real:
        pad = np.full((he - he_real, *img.shape[1:]), cval, dtype=img.dtype)
        img = np.concatenate((img, pad), 0)
    if we > we_real:
        pad = np.full((img.shape[0], we - we_real, *img.shape[2:]), cval, dtype=img.dtype)
        img = np.concatenate((img, pad), 1)
    return img