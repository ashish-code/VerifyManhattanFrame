# updated 12/18/2017 -- Fixed scoping bug with certain python distros
# updated 12/06/2017 -- Now requires browser credentials

from __future__ import print_function

import os
import errno
import sys
import argparse
import json
import requests
import multiprocessing
from urllib import request


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise


def download(data, directory, sign_url, headers):
    if 'file_name' in data and 'media_type' in data:
        file_name = data['file_name']
        prefix    = "videos" if data['media_type'] == 'video' else "images" if data['media_type'] == 'image' else 'audio'
        response  = requests.get(sign_url + "?file=%s&prefix=%s" % (file_name, prefix), headers=headers)
        if response.status_code == requests.codes.ok:
            url = response.json()["url"]
        else:
            url = "https://s3.amazonaws.com/medifor/%s/%s" % (prefix, file_name)
    else:
        print("Metadata entries must contain the 'file_name' and 'media_type' fields to download.")
        return

    downloadFilename = os.path.join(os.path.join(directory, prefix), file_name)
    subdirs = file_name.split("/")
    if len(subdirs) > 1:
        root = os.path.join(directory, prefix)
        for i in range(0, len(subdirs)-1):
            mkdir_p(os.path.join(root, subdirs[i]))
            root = os.path.join(root, subdirs[i])

    if not os.path.exists(downloadFilename):
        display_url = url.split("?")[0] if "?" in url else url
        print("Downloading %s to %s" % (display_url, downloadFilename))
        request.urlretrieve(url, downloadFilename)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser(description="Download media from metadata file.")
    parser.add_argument("credentials",       help="User credentials, in format 'username:password'.")
    parser.add_argument("metadata_file",     help="Metadata for the images to download.")
    parser.add_argument("-d", "--directory", help="Directory to download media to.")
    parser.add_argument("-j", "--jobs",      help="Number of download processes to spawn.", type=int)
    parser.add_argument("-l", "--limit",     help="Limit the number of downloads.", type=int)
    args = parser.parse_args()

    # Login to get access token
    credentials = args.credentials.split(':')
    response = requests.post('https://medifor.rankone.io/api/login/', {'username': credentials[0], 'password': credentials[1]})
    if response.status_code == requests.codes.ok:
        token = response.json()['key']
    else:
        print("Bad credentials!")
        sys.exit(1)

    # set headers and sign url
    headers  = {"Content-Type": "application/json", "Authorization": "Token %s" % token}
    sign_url = "https://medifor.rankone.io/api/sign/"

    processCount = args.jobs if args.jobs else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processCount)

    directory = args.directory if args.directory else "."
    mkdir_p(os.path.join(directory, "images"))
    mkdir_p(os.path.join(directory, "videos"))
    mkdir_p(os.path.join(directory, "audio"))

    limit = args.limit if args.limit else sys.maxint

    with open(args.metadata_file, 'r') as infile:
        count = 0
        for data in infile:
            if count >= limit:
                break

            data = json.loads(data)
            pool.apply_async(download, [data, directory, sign_url, headers])
            count += 1

    pool.close()
    pool.join()

    print("Downloaded %d files" % count)