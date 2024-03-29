import pickle
import numpy as np
import time
from multiprocessing import Process, Queue
import cv2

from .game import Game, WORLD_MARGIN, generate_elevation


class MultiGame:
    def __init__(self, nworlds_h=1, nworlds_w=2, map_h=200, map_w=200, default_scale=4, min_render_time=0.01, seed=42, map_seed=42, save_period=-1):
        self.map_h = map_h
        self.map_w = map_w
        self.nworlds_h = nworlds_h
        self.nworlds_w = nworlds_w
        self.render_h = nworlds_h * map_h * default_scale + nworlds_h * 2 * WORLD_MARGIN
        self.render_w = nworlds_w * map_w * default_scale + nworlds_w * 2 * WORLD_MARGIN
        self.min_render_time = min_render_time
        self.save_period = save_period

        self.info = {}
        self.last_render_time = time.time()
        self.steps_count = None

        # Create global elevation map
        elevation = generate_elevation(nworlds_h * map_h, nworlds_w * map_w, map_seed)

        # Create worlds
        self.worlds = []
        for i in range(self.nworlds_h):
            for j in range(self.nworlds_w):
                idx = i * self.nworlds_w + j
                el = elevation[i * map_h: (i + 1) * map_h, j * map_w: (j + 1) * map_w]
                w = Game(map_h, map_w, default_scale, seed=seed + idx, subworld_id=idx, elevation=el, save_period=-1)
                self.worlds.append(w)

        self.games_started = False
        self.steps_count = 0


    def render(self, camera_x=None, camera_y=None, scale=4):
        image = np.zeros([self.render_h, self.render_w, 3], dtype='uint8')

        if not self.games_started:
            return image

        y = 0
        for i in range(self.nworlds_h):
            x = 0
            for j in range(self.nworlds_w):
                idx = i * self.nworlds_w + j
                # Get image
                img = self.worlds_info[idx]['render']
                h, w, _ = img.shape
                # Draw image
                image[y: y + h, x: x + w] = img
                x = x + w
            y = y + h

        # Write some Text
        cv2.putText(image, f'Step {self.steps_count}', (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, 2)

        end = time.time()
        time_overhead = self.min_render_time - (end - self.last_render_time)
        if time_overhead > 0:
            time.sleep(time_overhead)
        self.last_render_time = end
        return image


    def step(self):
        if not self.games_started:
            self.start_games()

        # Get info from queues
        for i in range(len(self.queues)):
            self.worlds_info[i] = self.queues[i].get()
            while self.queues[i].qsize() > 0:
                self.worlds_info[i] = self.queues[i].get()

        # Update steps count
        self.steps_count += 1

        # Save world restart file
        if self.save_period > 0 and self.steps_count % self.save_period == 0:
            self.stop_games()
            fname = f'worlds_{self.steps_count:08n}.pickle'
            with open(fname, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Saved as', fname)
            self.start_games()


    def start_games(self):
        self.create_portals()

        def game_loop(world, queue, signals):
            while True:
                # When stop signal is recieved, send world to the main process and exit
                if not signals.empty():
                    s = signals.get()
                    if s == 'stop':
                        world.destroy_portals()
                        queue.put({'world': world})
                        break
                # Physics and render step
                world.step()
                queue.put({'render': world.render(), 'steps_count': world.steps_count})

        self.processes = []
        self.queues = []
        self.signals = []
        self.worlds_info = []
        for w in self.worlds:
            # Copy sub-world to a separate process
            q = Queue(maxsize=10)
            s = Queue(maxsize=10)
            p = Process(target=game_loop, args=(w, q, s))
            # Start process
            p.start()
            self.processes.append(p)
            self.queues.append(q)
            self.signals.append(s)
            self.worlds_info.append({'render': w.render()})

        self.games_started = True


    def stop_games(self):
        self.games_started = False

        # Send stop signal to all processes
        for s in self.signals:
            s.put('stop')

        # Get sub-worlds current states from all processes
        for i in range(len(self.worlds)):
            while True:
                info = self.queues[i].get()
                if 'world' in info:
                    break
            self.worlds[i] = info['world']

        # Wait for processes to end
        for p in self.processes:
            p.join()

        # Clear variables
        self.processes = []
        self.queues = []
        self.signals = []
        self.worlds_info = []


    def spawn(self, creature=None, world_id=None, pos_x=None, pos_y=None, angle=None):
        assert not self.games_started
        if world_id is None:
            world_id = np.random.randint(len(self.worlds))
        w = self.worlds[world_id]
        w.spawn(creature, pos_x, pos_y, angle)

    def create_portals(self):
        for i in range(self.nworlds_h):
            for j in range(self.nworlds_w):
                # Left-right portals
                if j < self.nworlds_w - 1:
                    idx_left = i * self.nworlds_w + j
                    idx_right = i * self.nworlds_w + j + 1
                    q_lr, q_rl = Queue(), Queue()
                    self.worlds[idx_left].add_portal('right', q_rl, q_lr)
                    self.worlds[idx_right].add_portal('left', q_lr, q_rl)
                # Up-down portals
                if i < self.nworlds_h - 1:
                    idx_up = i * self.nworlds_w + j
                    idx_down = (i + 1) * self.nworlds_w + j
                    q_ud, q_du = Queue(), Queue()
                    self.worlds[idx_up].add_portal('down', q_du, q_ud)
                    self.worlds[idx_down].add_portal('up', q_ud, q_du)
