"""Make the extension zip for one platform: the add-on plus that platform's renderer library.

    python tools/package.py --lib path/to/libspirula_render.so --platform linux-x64

Blender extension zips have the add-on's files at the top level; the manifest's `platforms`
says which system the zip installs on, so each platform gets its own zip.
"""

import argparse
import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "spirula_render")
LIB_NAMES = {"macos-arm64": "libspirula_render.dylib", "linux-x64": "libspirula_render.so",
             "windows-x64": "spirula_render.dll"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="the built renderer library")
    ap.add_argument("--platform", required=True, choices=sorted(LIB_NAMES))
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    manifest = open(os.path.join(PKG, "blender_manifest.toml")).read()
    version = re.search(r'^version = "([^"]+)"', manifest, re.M).group(1)
    manifest = re.sub(r'^platforms = .*$', f'platforms = ["{a.platform}"]', manifest, flags=re.M)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"spirula_render-{version}-{a.platform}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("blender_manifest.toml", manifest)
        for name in sorted(os.listdir(PKG)):
            if name.endswith((".py", ".json")):
                z.write(os.path.join(PKG, name), name)
        z.write(a.lib, "lib/" + LIB_NAMES[a.platform])
    print(path)


if __name__ == "__main__":
    main()
