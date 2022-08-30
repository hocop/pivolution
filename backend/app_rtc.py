"""my_controller controller."""

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
from av import VideoFrame

# webrtc imports
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.rtcrtpsender import RTCRtpSender
# from aiortc.contrib.signaling import BYE, ApprtcSignaling

from aiohttp import web

from pivolution.game import Game

ROOT = os.path.dirname(__file__)


class KeyQueue:
    def __init__(self, dt=0.1):
        self._lst = []
        self.dt = dt

    def append(self, key):
        self._lst.append((key, time.time()))
        self._clear()

    def _clear(self):
        self._lst = [el for el in self._lst if time.time() - el[1] < self.dt]

    def get_list(self):
        self._clear()
        return [el[0] for el in self._lst]


camera_image = np.zeros([720, 1280, 3], dtype='uint8')
pressed_keys = KeyQueue()
pcs = set()

class VideoImageTrack(VideoStreamTrack):
    """
    A video stream track that returns an image from robot camera.
    """

    def __init__(self):
        super().__init__()
        self.frames_count = 0

    async def recv(self):
        global camera_image
        pts, time_base = await self.next_timestamp()

        # create video frame
        frame = VideoFrame.from_ndarray(camera_image[:, :, ::-1], format="bgr24")
        frame.pts = pts
        frame.time_base = time_base

        return frame

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    print(codecs)
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)

    @pc.on("datachannel")
    def on_datachannel(channel):
        #channel_log(channel, "-", "created by remote party")

        @channel.on("message")
        def on_message(message):
            # print(channel.label, "<", message)

            if isinstance(message, str) and message.startswith("ping"):
                # reply
                channel.send("pong" + message[4:])
            
            if 'ArrowUp' in message:
                pressed_keys.append(ord('W'))
            if 'ArrowLeft' in message:
                pressed_keys.append(ord('A'))
            if 'ArrowDown' in message:
                pressed_keys.append(ord('S'))
            if 'ArrowRight' in message:
                pressed_keys.append(ord('D'))

    # Add video track
    video = VideoImageTrack()
    video_sender = pc.addTrack(video)
    # force_codec(pc, video_sender, 'video/H264')

    # Add recieving video track
    @pc.on("track")
    def on_track(track):
        print("Track %s received" % track.kind)
#        recorder.addTrack(track)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def main(args):
    global camera_image

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
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=args.host, port=args.port, ssl_context=ssl_context)
    await site.start()

    game = Game()
    for i in range(1000):
        game.spawn()
    # Main loops:
    def render_loop():
        global camera_image
        while True:
            camera_image = game.render()
    def sim_loop():
        while True:
            game.step()
    threading.Thread(target=render_loop).start()
    threading.Thread(target=sim_loop).start()

    while True:
        await asyncio.sleep(3600)  # sleep forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--play-from", help="Read the media from a file and sent it."),
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    asyncio.run(main(args))
