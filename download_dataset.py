#!/usr/bin/env python3
"""
Download script for the Miller ECoG Library (2019).

Downloads all dataset files from the Stanford Digital Repository
(https://purl.stanford.edu/zk881ps0522) and optionally unzips them.

Usage:
    python download_dataset.py                  # Download and unzip all files
    python download_dataset.py --continuous     # Only the 3 continuous decoding tasks (~830 MB)
    python download_dataset.py --no-unzip       # Download only, skip unzipping
    python download_dataset.py --output ./data  # Download to a custom directory
    python download_dataset.py --select         # Interactively select files
    python download_dataset.py --verify-only    # Verify existing downloads

Requires: Python 3.6+ (no external dependencies)
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error
import zipfile

BASE_URL = "https://stacks.stanford.edu/file/druid:zk881ps0522"

# Filenames for the three continuous motor decoding tasks used in the benchmark.
CONTINUOUS_FILENAMES = {
    "fingerflex.zip",
    "joystick_track.zip",
    "mouse_track.zip",
}

# All files from the Stanford Digital Repository with their sizes and MD5 checksums.
FILES = [
    {"filename": "BCI_Competion4_dataset4_data_fingerflexions.zip", "size": 232911694, "md5": "b67f5087f0582144b66104aa60387020"},
    {"filename": "MANUSCRIPT_kjm_ECoGLibrary_v14_FiguresIncluded.pdf", "size": 8999316, "md5": "f07e0f7a1bcc96007833e71eb68f1a43"},
    {"filename": "ctmr.zip", "size": 101836820, "md5": "d7de2749abcffc2a094a0455c547acf2"},
    {"filename": "faces_basic.zip", "size": 613144696, "md5": "812d92b2be0de466ba6c6683b53ea7f8"},
    {"filename": "faces_noise.zip", "size": 938380820, "md5": "89271b2993b09c804f78b43fe92b47b0"},
    {"filename": "fingerflex.zip", "size": 580579128, "md5": "8e44fef2a12ef42868b01a031f5b6350"},
    {"filename": "fixation_PAC.zip", "size": 214897262, "md5": "53bb098991aae4bb3cccc8e7544ac7ff"},
    {"filename": "fixation_highfreq.zip", "size": 349585584, "md5": "9ef3ae01bd10fa318fabe7f1671ba039"},
    {"filename": "fixation_pwrlaw.zip", "size": 388331766, "md5": "6c9c00009eb38f2fd9d04c6575dd6663"},
    {"filename": "gestures.zip", "size": 842016464, "md5": "30de1a74cab10feb2c13e0591df31462"},
    {"filename": "imagery_basic.zip", "size": 570200216, "md5": "38981b0a728ab96b42f65ff784e197ef"},
    {"filename": "imagery_feedback.zip", "size": 266345448, "md5": "dbf271d4b18becc0a354a718073ce5a0"},
    {"filename": "joystick_track.zip", "size": 141590618, "md5": "08c54cfaffae83ccbd3ec5dbdeeb2382"},
    {"filename": "kjm_ECoGLibrary_PatientTaskTable.pdf", "size": 116825, "md5": "a320c5dbfc826fd390317e45426fef27"},
    {"filename": "loc.zip", "size": 69882040, "md5": "0a4d8c4b447cceacc5e2c8f546ed3f4f"},
    {"filename": "memory_nback.zip", "size": 547718236, "md5": "408e98090a93c9f0ee58ac07c0b5a339"},
    {"filename": "motor_basic.zip", "size": 855006254, "md5": "cf87d50299b950ba251cf210217bb0aa"},
    {"filename": "mouse_track.zip", "size": 121933740, "md5": "28fa00d57f89ec1a289292c8362faf25"},
    {"filename": "speech_basic.zip", "size": 205616052, "md5": "3b7c52b532dbea4d7a98bfd0ce2576ca"},
    {"filename": "speech_lists.zip", "size": 581368620, "md5": "5557d752a26ecfe4ca6a649e9ccdfb7c"},
    {"filename": "toolbox.zip", "size": 135801, "md5": "d8ddb9e273e26eba742d89222e53b539"},
    {"filename": "visual_search.zip", "size": 161710891, "md5": "2a85384b8fe58e850d0bf724d08109d1"},
    {"filename": "xs_files.zip", "size": 21682022, "md5": "10918f5af3c4258d66068f583f41447b"},
]


def fmt_size(num_bytes):
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def md5_file(path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url, dest, expected_size=None):
    """Download a file with a progress bar (no dependencies)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Miller-ECoG-Downloader/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", expected_size or 0))
        downloaded = 0
        block_size = 1024 * 256  # 256 KB

        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = downloaded / total * 100
                    bar_len = 40
                    filled = int(bar_len * downloaded // total)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}%  {fmt_size(downloaded)} / {fmt_size(total)}"
                    )
                else:
                    sys.stdout.write(f"\r  Downloaded {fmt_size(downloaded)}")
                sys.stdout.flush()

    sys.stdout.write("\n")
    return downloaded


def unzip_file(zip_path, dest_dir):
    """Extract a zip file to the destination directory."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def select_files(files):
    """Interactive file selection."""
    print("\nAvailable files:\n")
    for i, f in enumerate(files):
        print(f"  [{i + 1:2d}] {f['filename']:55s} ({fmt_size(f['size'])})")

    print(f"\n  [ 0] Download ALL files ({fmt_size(sum(f['size'] for f in files))} total)")
    print()

    while True:
        choice = input("Enter file numbers separated by commas (e.g. 1,3,5) or 0 for all: ").strip()
        if not choice:
            continue
        try:
            nums = [int(x.strip()) for x in choice.split(",")]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
            continue

        if 0 in nums:
            return files

        if all(1 <= n <= len(files) for n in nums):
            return [files[n - 1] for n in nums]

        print(f"Please enter numbers between 0 and {len(files)}.")


def main():
    parser = argparse.ArgumentParser(
        description="Download the Miller ECoG Library from the Stanford Digital Repository."
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory for downloaded files (default: ./data)",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Skip unzipping downloaded .zip files",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Download only the 3 continuous decoding datasets: fingerflex, joystick_track, mouse_track (~830 MB)",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Interactively select which files to download",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify MD5 checksums of existing files, do not download",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip MD5 verification after download",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep .zip files after extraction (by default they are kept)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    files = FILES
    if args.continuous:
        files = [f for f in FILES if f["filename"] in CONTINUOUS_FILENAMES]
    if args.select:
        files = select_files(files)

    total_size = sum(f["size"] for f in files)
    zip_files = [f for f in files if f["filename"].endswith(".zip")]
    pdf_files = [f for f in files if f["filename"].endswith(".pdf")]

    print(f"\nMiller ECoG Library Downloader")
    print(f"Source: https://purl.stanford.edu/zk881ps0522")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(files)} ({fmt_size(total_size)})")
    print()

    # --- Verify-only mode ---
    if args.verify_only:
        print("Verifying existing files...\n")
        ok, fail, missing = 0, 0, 0
        for f in files:
            path = os.path.join(output_dir, f["filename"])
            if not os.path.isfile(path):
                print(f"  MISSING  {f['filename']}")
                missing += 1
                continue
            digest = md5_file(path)
            if digest == f["md5"]:
                print(f"  OK       {f['filename']}")
                ok += 1
            else:
                print(f"  FAIL     {f['filename']}  (expected {f['md5']}, got {digest})")
                fail += 1
        print(f"\nResults: {ok} OK, {fail} failed, {missing} missing")
        sys.exit(1 if fail > 0 else 0)

    # --- Download ---
    downloaded_files = []
    skipped = 0

    for i, f in enumerate(files, 1):
        dest = os.path.join(output_dir, f["filename"])
        url = f"{BASE_URL}/{urllib.request.quote(f['filename'])}"
        label = f"[{i}/{len(files)}]"

        # Skip if already downloaded and correct size
        if os.path.isfile(dest) and os.path.getsize(dest) == f["size"]:
            print(f"{label} {f['filename']} - already exists, skipping")
            downloaded_files.append(dest)
            skipped += 1
            continue

        print(f"{label} Downloading {f['filename']} ({fmt_size(f['size'])})")
        try:
            download_file(url, dest, expected_size=f["size"])
        except (urllib.error.URLError, OSError) as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping {f['filename']}. You can retry later.")
            continue

        # Verify MD5
        if not args.skip_verify:
            sys.stdout.write("  Verifying MD5... ")
            sys.stdout.flush()
            digest = md5_file(dest)
            if digest == f["md5"]:
                print("OK")
            else:
                print(f"MISMATCH (expected {f['md5']}, got {digest})")
                print("  WARNING: File may be corrupted. Consider re-downloading.")

        downloaded_files.append(dest)

    # --- Unzip ---
    if not args.no_unzip:
        zip_downloads = [p for p in downloaded_files if p.endswith(".zip")]
        if zip_downloads:
            print(f"\nExtracting {len(zip_downloads)} zip files...\n")
            for zp in zip_downloads:
                name = os.path.basename(zp)
                extract_dir = output_dir
                sys.stdout.write(f"  Extracting {name}... ")
                sys.stdout.flush()
                try:
                    unzip_file(zp, extract_dir)
                    print("OK")
                except zipfile.BadZipFile:
                    print("ERROR (bad zip file)")

    # --- Summary ---
    print(f"\nDone!")
    print(f"  Downloaded: {len(downloaded_files) - skipped}")
    print(f"  Skipped (already existed): {skipped}")
    if not args.no_unzip:
        print(f"  Extracted: {len([p for p in downloaded_files if p.endswith('.zip')])}")
    print(f"  Location: {output_dir}")


if __name__ == "__main__":
    main()
