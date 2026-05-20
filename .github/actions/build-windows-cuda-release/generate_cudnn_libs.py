"""Generate .lib import libraries from nvidia-cudnn-cu12 pip package DLLs.

The nvidia-cudnn-cu12 pip package ships headers (.h) and runtime DLLs
but no MSVC import libraries (.lib). CMake's FindCUDNN needs .lib files
to link against. This script generates them using dumpbin + lib (MSVC tools).

Writes CUDNN_INCLUDE_PATH and CUDNN_LIBRARY_PATH to GITHUB_ENV.
"""

import importlib.metadata
import os
import pathlib
import re
import subprocess
import sys


def find_cudnn_package():
    dist = importlib.metadata.distribution("nvidia-cudnn-cu12")
    root = pathlib.Path(dist._path).parent
    include_dir = next(root.rglob("cudnn.h"), None)
    bin_dir = next(root.rglob("cudnn64_9.dll"), None)
    if not include_dir or not bin_dir:
        print("ERROR: Could not locate cuDNN headers or DLLs", file=sys.stderr)
        sys.exit(1)
    return include_dir.parent, bin_dir.parent


def generate_import_lib(dll_path, lib_dir):
    lib_name = re.sub(r"64_\d+$", "", dll_path.stem)
    result = subprocess.run(
        ["dumpbin", "/exports", str(dll_path)],
        capture_output=True,
        text=True,
    )
    exports = []
    for line in result.stdout.splitlines():
        m = re.match(r"\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)", line)
        if m:
            exports.append(m.group(1))

    if not exports:
        print(f"  WARNING: No exports found in {dll_path.name}, skipping")
        return

    def_path = lib_dir / f"{lib_name}.def"
    def_path.write_text(
        f"LIBRARY {dll_path.name}\nEXPORTS\n"
        + "\n".join(f"  {e}" for e in exports)
        + "\n"
    )

    lib_path = lib_dir / f"{lib_name}.lib"
    subprocess.check_call(
        ["lib", f"/def:{def_path}", f"/out:{lib_path}", "/machine:x64"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  Generated {lib_name}.lib ({len(exports)} exports)")


def main():
    include_dir, bin_dir = find_cudnn_package()
    lib_dir = bin_dir.parent / "lib" / "x64"
    lib_dir.mkdir(parents=True, exist_ok=True)

    print(f"cuDNN include: {include_dir}")
    print(f"cuDNN DLLs:    {bin_dir}")
    print(f"cuDNN libs:    {lib_dir}")

    dlls = sorted(bin_dir.glob("cudnn*.dll"))
    print(f"\nGenerating import libraries for {len(dlls)} DLLs:")
    for dll in dlls:
        generate_import_lib(dll, lib_dir)

    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"CUDNN_INCLUDE_PATH={include_dir}\n")
            f.write(f"CUDNN_LIBRARY_PATH={lib_dir}\n")
        print(f"\nWrote CUDNN_INCLUDE_PATH and CUDNN_LIBRARY_PATH to GITHUB_ENV")
    else:
        print(f"\nCUDNN_INCLUDE_PATH={include_dir}")
        print(f"CUDNN_LIBRARY_PATH={lib_dir}")


if __name__ == "__main__":
    main()
