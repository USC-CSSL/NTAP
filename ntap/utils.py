import requests
from tqdm import tqdm

def download_file(url, local_dest):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size= 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(local_dest, 'wb') as fo:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            fo.write(data)
    progress_bar.close()
    #if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        #self.log.error(f"Could not download {url} to {local_dest}")

#def load_imdb(file_name):
    #d = {'pos': list(), 'neg': list()}
    #zf = zipfile.ZipFile(file_name, 'r')
    #for name in zf.namelist():
        #if name.startswith('pos') and name.endswith('txt'):
            #data = zf.read(name)
            #d['pos'].append(data.strip())
        #if name.startswith('neg') and name.endswith('txt'):
            #data = zf.read(name)
            #d['neg'].append(data.strip())
    #return d
