"""Build the optional Python entry accelerator from source; no downloaded binaries.

    python src/build_hotstep.py

Windows requires Visual Studio C++ tools. Linux/macOS require a C compiler.
No fast-math: parity with the reference is part of the validation contract.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def build():
    env = os.environ.copy()
    if sys.platform == "win32" and not shutil.which("cl"):
        vswhere = Path(env.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        install = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
        setup = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
        if not setup.is_file():
            raise RuntimeError("MSVC C++ tools not installed")
        command = f'call "{setup}" >nul && set'
        env["PATH"] = str(vswhere.parent) + os.pathsep + env.get("PATH", "")
        # cmd uses its own quoting, not the backslash-quote escaping of list2cmdline.
        output = subprocess.check_output('cmd.exe /d /s /c "' + command + '"', text=True, env=env)
        env.update(dict(line.split("=", 1) for line in output.splitlines() if "=" in line and not line.startswith("=")))
    suffix = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    dest = ROOT / ("hotstep" + suffix)
    with tempfile.TemporaryDirectory(prefix="pathfinder-hot-build-") as tmp:
        binary = Path(tmp) / dest.name
        if sys.platform == "win32":
            compiler = shutil.which("cl", path=env.get("PATH", env.get("Path", "")))
            if not compiler:
                raise RuntimeError("MSVC initialization did not supply cl.exe")
            command = [compiler, "/nologo", "/O2", "/fp:precise", "/LD", str(ROOT / "hotstep.c"), f"/Fe:{binary}"]
        else:
            command = [os.environ.get("CC", "cc"), "-O2", "-std=c11", "-Wall", "-Wextra", "-ffp-contract=off",
                       "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared",
                       str(ROOT / "hotstep.c"), "-o", str(binary), "-lm"]
        subprocess.run(command, cwd=tmp, env=env, check=True)
        shutil.copyfile(binary, dest)
    metadata = {"abi": 1, "source_sha256": hashlib.sha256((ROOT / "hotstep.c").read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "platform": platform.platform(), "command": command}
    dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    build()
