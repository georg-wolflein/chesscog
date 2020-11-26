import urllib.request
import zipfile
from recap import URI


zip_file = URI("data://games.zip")
urllib.request.urlretrieve("https://www.pgnmentor.com/players/Carlsen.zip",
                           zip_file)
with zipfile.ZipFile(zip_file) as zip_f:
    with zip_f.open("Carlsen.pgn", "r") as in_f, URI("data://games.pgn").open("wb") as out_f:
        out_f.write(in_f.read())
