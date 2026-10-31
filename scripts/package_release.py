"""Create an allowlisted source archive, excluding local data and historical artifacts."""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "requirements-d4rl.txt",
    ".gitignore",
    "MANIFEST.in",
    "train.py",
    "evaluate.py",
)
DIRECTORIES = ("qdc", "scripts", "configs", "docs", "tests", "third_party", "assets", ".github")
SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".png"}


def source_files():
    for name in FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is missing: {path}")
        yield path
    for name in DIRECTORIES:
        directory = ROOT / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Required release directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts:
                yield path


def main():
    output = ROOT / "dist" / "qdc-source.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, Path("qdc") / path.relative_to(ROOT))
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    print(f"Created {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
