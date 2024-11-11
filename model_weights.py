import os
import requests
from tqdm import tqdm
import gzip

def download_word2vec(destination_path='GoogleNews-vectors-negative300.bin.gz'):
    """
    Downloads the Google News pre-trained Word2Vec model.
    """
    url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
    chunk_size = 1024

    destination_dir = os.path.dirname(destination_path)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)

    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            raise ValueError(f"Failed to download file from {url}, status code: {response.status_code}")

        with open(destination_path, 'wb') as file:
            total = response.headers.get('content-length')
            total = int(total) if total is not None else None
            
            with tqdm(total=total, unit='B', unit_scale=True, desc=destination_path) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    progress_bar.update(len(chunk))
    
    print("Download complete!")

def extract_word2vec(file_path, extract_to='GoogleNews-vectors-negative300.bin'):

    with open(file_path, 'rb') as f:
        magic_number = f.read(2)
    
    if magic_number == b'\x1f\x8b': 
        print("Extracting gzipped file...")
        with gzip.open(file_path, 'rb') as f_in:
            with open(extract_to, 'wb') as f_out:
                f_out.write(f_in.read())
        print("Extraction complete!")
    else:
        print("File is not gzipped. No extraction needed.")
        os.rename(file_path, extract_to)  

word2vec_gz_path = 'GoogleNews-vectors-negative300.bin.gz'
word2vec_bin_path = 'GoogleNews-vectors-negative300.bin'

download_word2vec(destination_path=word2vec_gz_path)
extract_word2vec(file_path=word2vec_gz_path, extract_to=word2vec_bin_path)