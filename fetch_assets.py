"""Download ONLY the pinned Panda model and its referenced meshes, with licenses."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
import urllib.request
import xml.etree.ElementTree as ET

COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
ROOT = Path(__file__).resolve().parent
DEST = ROOT / "vendor" / "franka_emika_panda"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/franka_emika_panda/"


def fetch(name):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(BASE + name, timeout=60) as response:
            blob = response.read()
        path.write_bytes(blob)
    return name, sha256(path.read_bytes()).hexdigest()


def main():
    manifest = dict(fetch(name) for name in ["panda_nohand.xml", "LICENSE", "README.md"])
    tree = ET.parse(DEST / "panda_nohand.xml")
    files = ["assets/" + node.attrib["file"] for node in tree.findall("./asset/mesh")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        manifest.update(pool.map(fetch, files))
    (DEST / "manifest.json").write_text(json.dumps({"commit": COMMIT, "sha256": manifest}, indent=2) + "\n")
    print(f"Panda: {len(manifest)} files at {COMMIT}; {DEST}")


if __name__ == "__main__":
    main()
