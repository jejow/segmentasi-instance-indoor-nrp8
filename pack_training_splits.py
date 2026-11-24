"""Copy byte-identical official train and validation members to a training archive."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


def pack(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Use a separate path for the train and validation archive.")
    with source.open("rb") as stream:
        source_sha256 = file_digest(stream, "sha256").hexdigest()
    with ZipFile(source) as original:
        manifest = json.loads(original.read("manifest.json"))
        subset = {key: manifest["splits"][key] for key in ("train", "valid")}
        if [subset[key]["images"] for key in ("train", "valid")] != [397, 166]:
            raise ValueError("Expected the official Panoptic Indoor v5 train and validation splits.")
        if destination.is_file():
            with ZipFile(destination) as existing:
                receipt = json.loads(existing.read("manifest.json"))
            if receipt["parent_archive_sha256"] != source_sha256 or receipt["splits"] != subset:
                raise ValueError("Existing train and validation archive has different provenance.")
            return
        manifest.update(splits=subset, parent_archive_sha256=source_sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".part")
        with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=3) as packed:
            packed.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
            for name in original.namelist():
                if name.startswith(("train/", "valid/")):
                    method = ZIP_STORED if name.lower().endswith(".jpg") else ZIP_DEFLATED
                    packed.writestr(name, original.read(name), compress_type=method)
        partial.replace(destination)
    print(f"{destination}: {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    options = parser.parse_args()
    pack(options.source, options.destination)
