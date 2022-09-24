"""my_controller controller."""

import pickle
import numpy as np
import argparse
import asyncio
import json
import logging
import os
import platform
import ssl
import time
import threading
import cv2

from PIL import Image
from av import VideoFrame

from aiohttp import web
from io import BytesIO

from pivolution.creatures import CreatureRecurrent, CreatureGendered
from pivolution.game import Game
from pivolution.multigame import MultiGame


camera_image = np.zeros([100, 100, 3], dtype='uint8')

FPS = 10
last_frame_time = -1

out_video = None

async def frame(request):
    # camera_image = game.render()
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
        game = MultiGame(2, 2, map_h=120, map_w=200)
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
        global last_frame_time
        while True:
            game.step()
            camera_image = game.render()

            # Write to output file
            if args.write_video is not None:
                t = game.steps_count
                if t is not None and t > last_frame_time:
                    out_video.write(camera_image[:, :, ::-1])
                    last_frame_time = t

    threading.Thread(target=game_loop).start()

    while True:
        await asyncio.sleep(3600)  # sleep forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--load_from", type=str, default=None, help="Load game from restart file")
    parser.add_argument("--write_video", type=str, default=None, help="Path to write video")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print('#' * 100)
        print('Finishing')
        if args.write_video is not None:
            out_video.release()
        print('Finished')
        print('#' * 100)
