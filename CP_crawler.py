# must run with python3.6 or above
from __future__ import absolute_import
import os
import requests
from subprocess import Popen, PIPE
import re

home_folder_name = r"/home/jambrown/" # TODO change this to your home folder
start_date = 20210630   # in the form of yyyymmdd, e.g., 20210101 # TODO change this to your preferred date
data_dir = home_folder_name +r"CP_Downloads/"

def main():
    fetched_files = set(os.listdir(data_dir))

    file_name_prefixes = ["CP_Satellite"]
    url = "https://storage.googleapis.com/censoredplanetscanspublic/"
    # crawl new files
    for tech in file_name_prefixes:
        print(tech)
        sat_url = 'https://data.censoredplanet.org/raw?technique='+tech
        r = requests.get(sat_url).content
        files = re.findall(tech+'-[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+.tar.gz', str(r))
        for file_name in files:
            print(file_name)
            file_uploaded_date = file_name.split("-")[-6:-3]
            if file_name not in fetched_files and \
                    int(start_date) <= int(''.join(file_uploaded_date)):
                Popen("wget {0}{1} -P {2}; chown :cdac {1}".format(url, file_name, data_dir),
                      stdout=PIPE, shell=True).wait()

if __name__ == "__main__":
  main()