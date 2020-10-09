"""Script to download the rendered dataset."""

from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile
import shutil
from pathlib import Path

from chesscog import DATA_DIR

render_dir = DATA_DIR / "render"
zip_file = DATA_DIR / "render.zip"
print("Downloading dataset...")
gdd.download_file_from_google_drive(file_id="1fTX22T5nMjwzJBy228yEESapAY5TEVII",
                                    dest_path=zip_file,
                                    overwrite=True,
                                    showsize=True)

print("Unzipping dataset...")
shutil.rmtree(render_dir, ignore_errors=True)
with zipfile.ZipFile(zip_file, "r") as f:
    f.extractall(path=render_dir.parent)

print(f"Downloaded dataset to {render_dir}.")
