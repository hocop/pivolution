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

from PIL import Image
from av import VideoFrame

from aiohttp import web
from io import BytesIO

from pivolution.creature import CreatureGendered
from pivolution.game import Game


camera_image = np.zeros([100, 100, 3], dtype='uint8')


async def frame(request):
    # camera_image = game.render()
    stream = BytesIO()
    Image.fromarray(camera_image).save(stream, "PNG")
    return web.Response(body=stream.getvalue(), content_type='image/PNG')



async def main(args):
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
        game = Game()
        # Spawn initial population
        for i in range(5000):
            game.spawn()

        # for i in range(0, game.map_h, 5):
        #     for j in range(0, game.map_w, 3):
        #         creature0 = CreatureGendered()
        #         for pos_y in range(i, i + 2):
        #             for pos_x in range(j, j + 1):
        #                 gender = 'boy' if pos_x % 2 == 0 else 'girl'
        #                 creature = CreatureGendered(genes=creature0.genes.copy(), gender=gender)
        #                 game.spawn(creature, pos_x, pos_y)

    # Main loops:
    def render_loop():
        global camera_image
        while True:
            game.step()
            camera_image = game.render()
    def sim_loop():
        while True:
            game.step()
    threading.Thread(target=render_loop).start()
    # threading.Thread(target=sim_loop).start()

    while True:
        await asyncio.sleep(3600)  # sleep forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--load_from", type=str, default=None, help="Load game from restart file")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    asyncio.run(main(args))
