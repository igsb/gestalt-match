#!/usr/bin/env python3
'''
Download the photo and meta data
'''
import configparser
from lib.face2gene import Face2Gene

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    print('--Init Configuration--')
    f2g = Face2Gene(config['Download'])
    print('--Start download--')
    f2g.download_photo_all()

if __name__ == '__main__':
    main()
