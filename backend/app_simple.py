"""my_controller controller."""
import os
import pickle
import numpy as np
import argparse
import asyncio
import logging
import ssl
import cv2
import threading

from PIL import Image
from av import VideoFrame

from aiohttp import web
from io import BytesIO

from pivolution.creatures import CreatureRecurrent, CreatureGendered
from pivolution.game import Game
from pivolution.multigame import MultiGame


# Make numpy use only one thread. One thread is much faster
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


camera_image = np.zeros([100, 100, 3], dtype='uint8')

FPS = 30

out_video = None

async def frame(request):
    stream = BytesIO()
    Image.fromarray(camera_image).save(stream, "PNG")
    return web.Response(body=stream.getvalue(), content_type='image/PNG')



async def main(args):
    global out_video

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None#ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

    app = web.Application()
    app.router.add_post("/frame", frame)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=args.host, port=args.port, ssl_context=ssl_context)
    await site.start()

    if args.load_from is not None:
        with open(args.load_from, 'rb') as handle:
            game = pickle.load(handle)
    else:
        # game = Game()
        # for _ in range(1000):
        #     creature = CreatureGendered()
        #     game.spawn(creature)

        game = MultiGame(2, 3, map_h=160, map_w=160)
        # Spawn initial population
        for i in range(game.nworlds_h):
            for j in range(game.nworlds_w):
                idx = i * game.nworlds_w + j
                for _ in range(1000):
                    creature = CreatureGendered() if (i + j) % 2 == 0 else CreatureRecurrent()
                    game.spawn(creature, idx)
        del creature

    # Open output video file
    if args.write_video is not None:
        out_video = cv2.VideoWriter(args.write_video, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (game.render_w, game.render_h))

    # Main loops:
    def game_loop():
        global camera_image
        count_0 = game.steps_count
        while True:
            try:
                game.step()
                camera_image = game.render()

                # Write to output file
                if args.write_video is not None:
                    out_video.write(camera_image[:, :, ::-1])

                if args.max_steps is not None and game.steps_count - count_0 > args.max_steps:
                    game.stop_games()
                    print('Max steps reached', game.steps_count, args.max_steps)
                    break
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                game.stop_games()
                break

    game_thread = threading.Thread(target=game_loop)
    game_thread.start()
    while game_thread.is_alive():
        await asyncio.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--load_from", type=str, default=None, help="Load game from restart file")
    parser.add_argument("--write_video", type=str, default=None, help="Path to write video")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to simulate")
    parser.add_argument("--save_period", type=int, default=100_000, help="Save every N steps")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    asyncio.run(main(args))
    print('#' * 100)
    print('Finishing')
    if args.write_video is not None:
        out_video.release()
    print('Finished')
    print('#' * 100)
