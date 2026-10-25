"""Generate through the actual HTTP API and validate the PNG response."""
import argparse
import io
import json
import urllib.request
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5732)
parser.add_argument("--size", default="512x512")
args = parser.parse_args()
base = f"http://127.0.0.1:{args.port}"
with urllib.request.urlopen(base+'/health',timeout=10) as response:
    state=json.load(response)
payload = {
    "prompt": "A small red fox resting in fresh snow beneath pine trees, natural winter light, detailed wildlife photograph.",
    "size": args.size, "steps": state["capabilities"]["default_steps"], "seed": 42, "response_format": "url",
}
request = urllib.request.Request(base + "/v1/images/generations", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=1800) as response:
    data = json.load(response)
url = data["data"][0]["url"]
with urllib.request.urlopen(url, timeout=30) as response:
    raw = response.read()
image = Image.open(io.BytesIO(raw))
image.load()
assert image.format == "PNG" and image.size == tuple(map(int, args.size.split("x")))
assert any(lo != hi for lo, hi in image.convert("RGB").getextrema()), "Image is blank"
print(json.dumps({"result": "PASS", "size": image.size, "seconds": data["generation_seconds"], "image": url}, indent=2))
