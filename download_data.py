import argparse
import numpy as np
import os

import Satellite_utils.CP_crawler as satellite_download

# from Satellite_utils.CP_crawler import main

## for Satellite use CP_crawler.py

def main():

    rng = np.random.default_rng(77)

    parser = argparse.ArgumentParser()

    parser.add_argument('--master_dir', type=str, default='/data/censorship/')
    parser.add_argument('--dataset', type=str, default='Satellite', choices=['Satellite', 'OONI'])
    parser.add_argument('--start_date', type=int, default=20220701)
    parser.add_argument('--end_date', type=int, default=20220731)

    args = parser.parse_args()

    if args.dataset=='Satellite':
        data_dir = args.master_dir + '/' + args.dataset + '/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        satellite_download.main(data_dir, args.start_date, args.end_date)
        # else:
        #     print('%s data has already been downloaded' % args.dataset)


if __name__ == "__main__":
  main()