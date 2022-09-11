import pickle
import queue
import numpy as np
import time
from multiprocessing import Process, Queue

from .creatures import CreatureRandom, CreatureLinear, CreatureNeural, CreatureRecurrent, CreatureGendered
from .game import Game


WORLD_MARGIN = 4


class MultiGame:
    def __init__(self, nworlds_h, nworlds_w, map_h=200, map_w=200, default_scale=4, min_render_time=0.01, seed=42, map_seed=42):
        self.map_h = map_h
        self.map_w = map_w
        self.nworlds_h = nworlds_h
        self.nworlds_w = nworlds_w
        self.render_h = (nworlds_h * map_h + (nworlds_h - 1) * WORLD_MARGIN) * default_scale
        self.render_w = (nworlds_w * map_w + (nworlds_w - 1) * WORLD_MARGIN) * default_scale
        self.min_render_time = min_render_time

        self.info = {}
        self.last_render_time = time.time()
        self.steps_count = None

        self.worlds = []
        for i in range(self.nworlds_h):
            for j in range(self.nworlds_w):
                idx = i * self.nworlds_w + j
                g = Game(map_h, map_w, default_scale, seed=seed + idx, map_seed=map_seed + idx, subworld_id=idx)
                self.worlds.append(g)

        self.games_started = False


    def render(self, camera_x=None, camera_y=None, scale=4):
        image = np.zeros([self.render_h, self.render_w, 3], dtype='uint8')

        if not self.games_started:
            return image

        y = 0
        for i in range(self.nworlds_h):
            x = 0
            for j in range(self.nworlds_w):
                idx = i * self.nworlds_w + j
                # Get image from queue
                if not self.queues[idx].empty():
                    self.worlds_info[idx] = self.queues[idx].get()
                # Draw image
                w = self.worlds[idx]
                image[y: y + w.render_h, x: x + w.render_w] = self.worlds_info[idx]['render']
                x = x + w.render_w + WORLD_MARGIN
            y = y + w.render_h + WORLD_MARGIN

        end = time.time()
        time_overhead = self.min_render_time - (end - self.last_render_time)
        if time_overhead > 0:
            time.sleep(time_overhead)
        self.last_render_time = end
        return image


    def step(self):
        if not self.games_started:
            self.start_games()

        # Update steps count
        self.steps_count = None
        counts = [info['steps_count'] for info in self.worlds_info if 'steps_count' in info]
        if len(counts) > 0:
            self.steps_count = min(counts) - 1

        # Save world restart file
        if self.steps_count is not None and self.steps_count % (3600 * 3) == 0:
            self.stop_games()
            fname = f'worlds_{self.steps_count:08n}.pickle'
            with open(fname, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Saved as', fname)
            self.start_games()


    def start_games(self):
        def game_loop(world, queue, signals):
            while True:
                if not signals.empty():
                    s = signals.get()
                    if s == 'stop':
                        queue.put({'world': world})
                        break
                world.step()
                queue.put({'render': world.render(), 'steps_count': world.steps_count})

        self.processes = []
        self.queues = []
        self.signals = []
        self.worlds_info = []
        for w in self.worlds:
            q = Queue(maxsize=10)
            s = Queue(maxsize=10)
            p = Process(target=game_loop, args=(w, q, s))
            p.start()
            self.processes.append(p)
            self.queues.append(q)
            self.signals.append(s)
            self.worlds_info.append({'render': w.render()})

        self.games_started = True


    def stop_games(self):
        self.games_started = False

        for s in self.signals:
            s.put('stop')

        for i in range(len(self.worlds)):
            w = None
            while w is None:
                info = self.queues[i].get()
                if 'world' in info:
                    w = info['world']
            self.worlds[i] = w

        for p in self.processes:
            p.join()

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
