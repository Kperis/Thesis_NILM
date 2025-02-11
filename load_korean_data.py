import os
import zipfile


def extract_zips(zip_dir = "datasets/korean/", extract_dir = "datasets/korean_extracted/"):
    os.makedirs(extract_dir, exist_ok=True)

    for zip_file in os.listdir(zip_dir):
        if zip_file.endswith(".zip"):
            house_id = zip_file.split(".")[0]
            house_extract_path = os.path.join(extract_dir, house_id)

        if not os.path.exists(house_extract_path):
            with zipfile.ZipFile(os.path.join(zip_dir, zip_file), 'r') as zip_ref:
                zip_ref.extractall(house_extract_path)

    print("Done extracting Korean data")

if __name__ == "__main__":
    extract_zips()