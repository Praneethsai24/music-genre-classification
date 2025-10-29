# Download GTZAN Dataset Script
import os
import requests
import zipfile

def download_gtzan(destination='data/gtzan.zip'):
    url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
    destination_tar = destination.replace('.zip', '.tar.gz')
    print("Downloading GTZAN dataset...")
    response = requests.get(url, stream=True)
    with open(destination_tar, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete. Extracting files...")
    import tarfile
    with tarfile.open(destination_tar) as tar:
        tar.extractall(path='data/')
    print("Done!")

if __name__ == "__main__":
    if not os.path.exists('data/genres'):
        download_gtzan()
    else:
        print('Dataset already exists.')