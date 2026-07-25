"""Convert AVIF files in an explicitly selected directory to PNG."""

import argparse
from pathlib import Path

import imageio.v3 as iio


def main(directory: Path, delete_source: bool = False) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    for source in directory.glob("*.avif"):
        destination = directory / f"{source.stem}.png"
        iio.imwrite(destination, iio.imread(source))
        print(f"Converted {source.name} -> {destination.name}")
        if delete_source:
            source.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--delete-source", action="store_true")
    arguments = parser.parse_args()
    main(arguments.directory, arguments.delete_source)
