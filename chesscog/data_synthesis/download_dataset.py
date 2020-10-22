"""Script to download the rendered dataset."""

from chesscog.utils.io import URI, download_zip_folder_from_google_drive

if __name__ == "__main__":
    download_zip_folder_from_google_drive("1XClmGJwEWNcIkwaH0VLuvvAY3qk_CRJh",
                                          URI("data://render"), show_size=True)
