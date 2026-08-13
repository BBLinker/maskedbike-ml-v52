#!/usr/bin/env python3
"""Build the deterministic BIKE v5.2 error replay helper from the pinned archive."""
from __future__ import annotations
import argparse
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

REFERENCE_SHA256 = "bf10eeff09c097ce011bef1f0d033815461102bb2484220aa7c75cd06caa6352"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-archive", required=True, type=Path)
    parser.add_argument("--helper-source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "native" / "bike_v52_error_reconstructor.cpp")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--openssl-include", type=Path)
    args = parser.parse_args()
    archive = args.reference_archive.resolve()
    if digest(archive) != REFERENCE_SHA256:
        raise ValueError("BIKE v5.2 reference archive SHA-256 differs from the capture provenance")
    with tempfile.TemporaryDirectory(prefix="bike-v52-error-build-") as temporary:
        build = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            source.extractall(build)
        implementation = build / "Reference_Implementation"
        kem = implementation / "kem.c"
        text = kem.read_text()
        include_anchor = '#include "shake_prng.h"\n'
        split_anchor = "    ntl_split_polynomial(e0, e1, e);\n"
        if text.count(split_anchor) != 1 or include_anchor not in text:
            raise RuntimeError("unexpected BIKE v5.2 kem.c layout")
        text = text.replace(include_anchor, include_anchor +
                            "\nuint8_t maskedbike_last_e0[R_SIZE] = {0};\n"
                            "uint8_t maskedbike_last_e1[R_SIZE] = {0};\n", 1)
        text = text.replace(split_anchor, split_anchor +
                            "    memcpy(maskedbike_last_e0, e0, R_SIZE);\n"
                            "    memcpy(maskedbike_last_e1, e1, R_SIZE);\n", 1)
        kem.write_text(text)
        helper = implementation / "bike_v52_error_reconstructor.cpp"
        shutil.copy2(args.helper_source, helper)
        sources = [helper, *sorted(implementation.glob("*.c")), implementation / "ntl.cpp",
                   implementation / "FromNIST" / "rng.c"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = ["g++", "-m64", "-O3", "-std=c++11", "-DNIST_RAND=1", "-DVERBOSE=0",
                   "-I", str(implementation)]
        if args.openssl_include is not None:
            include = args.openssl_include.resolve()
            command += ["-I", str(include)]
            multiarch = include / "x86_64-linux-gnu"
            if multiarch.is_dir():
                command += ["-I", str(multiarch)]
        system = Path("/lib/x86_64-linux-gnu")
        command += [*map(str, sources), str(system / "libcrypto.so.3"), str(system / "libssl.so.3"),
                    "-lm", "-ldl", "-lntl", str(system / "libgmp.so.10"),
                    str(system / "libgf2x.so.3"), "-lpthread", "-o", str(args.output)]
        subprocess.run(command, check=True)
    print(f"helper={args.output.resolve()} sha256={digest(args.output)}")


if __name__ == "__main__":
    main()
