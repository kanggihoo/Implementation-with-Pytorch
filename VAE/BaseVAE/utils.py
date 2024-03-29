import requests
import tarfile
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse
import os
 # https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
def download_data(
    url:str,
    data_path:str = "data",
    remove:bool = False
):
  # parser = urlparse(url)
  # url = parser.geturl()

  response = requests.get(url , stream=True)
  total_size = int(response.headers.get("content-length",0))
  block_size = 1024
  # tar_file = Path(__file__).parent / "download.tar"
  tar_file = Path.cwd().resolve() / "download.tar.gz"
  print("압축 파일 다운로드 시작")
  if not tar_file.exists():
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(tar_file , "wb") as file:
      for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)
    progress_bar.close()
    print("\n압축 파일 다운로드 완료")

  print("압축 파일 풀기 시작")
  extract_path = Path.cwd().resolve()/"data"
  if not extract_path.exists():
    extract_path.mkdir(parents=True , exist_ok=True)
  with open(tar_file , "rb") as file:
    with tarfile.open(fileobj=file,mode="r:gz" ) as tar:
      total_files = len(tar.getnames())
      progress_bar = tqdm(total=total_files, unit="files", unit_scale=True)
      for tarinfo in tar:
        progress_bar.update(1)
        tar.extract(tarinfo,path = extract_path)
      progress_bar.close()
  print("압축 파일 압축완료 ")
  if remove:
    os.remove(tar_file)



