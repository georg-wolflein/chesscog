import urllib.request
import zipfile
from chesscog import DATA_DIR

zip_file = DATA_DIR / "games.zip"
urllib.request.urlretrieve("https://www.pgnmentor.com/players/Carlsen.zip",
                           zip_file)
with zipfile.ZipFile(zip_file) as zip_f:
    with zip_f.open("Carlsen.pgn", "r") as in_f, (DATA_DIR / "games.pgn").open("wb") as out_f:
        out_f.write(in_f.read())
