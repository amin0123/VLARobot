"""Video annotations are only for human review; policy RGB remains unannotated."""

from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class VideoRecorder:
    def __init__(self, path, fps=10, scale=2):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "20", "preset": "fast"}
        self.scale = scale
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            self.font = ImageFont.load_default()

    def add(self, rgb, instruction="", status=""):
        import textwrap

        image = Image.fromarray(rgb).resize((rgb.shape[1] * self.scale, rgb.shape[0] * self.scale))
        canvas = Image.new("RGB", (image.width, image.height + 96), "#10151c")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        text = "\n".join(textwrap.wrap(instruction, width=58)) + "\n" + status
        draw.text((10, image.height + 8), text, font=self.font, fill="white")
        self.stream.width, self.stream.height = canvas.size
        frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if self.container is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            self.container = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
