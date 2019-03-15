import requests
from lxml import html
import os
from os import listdir
from os.path import isfile, join
import json
import numpy as np
import pandas as pd
import logging
import hashlib
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class Face2Gene(requests.Session):
    '''Login into Face2Gene to fetch patient info and photo
    '''

    login_url = 'https://app.face2gene.com/access/login'

    def __init__(self, config=None):
        super().__init__()
        logging.basicConfig(
            filename=os.path.join('log/download.log'),
            filemode='w',
            format='%(asctime)s: %(levelname)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )

        if config:
            user = config['user']
            password = config['password']
            self.project_dir, self.photo_dir, self.json_dir = self._init_path(config)
            self.input_path = config['input_path']

        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500])
        self.mount('http://', HTTPAdapter(max_retries=retries))
        self._connect(user, password)

    def download_photo_all(self):
        '''Download all photos of patients in the input folder
        Iterate all patients in input folder and download photo if
        patient has photo.
        '''
        case_list = [os.path.splitext(os.path.basename(f))[0] for f in listdir(self.input_path) if isfile(join(self.input_path, f))]

        photo_case_list = []
        no_photo_list = []
        data_list = []
        count = 0

        for case_id in case_list:
            url = "https://app.face2gene.com/cases/%s/images" % case_id
            response = self.get(url)
            if response.status_code == 200:
                photo_case_list.append(case_id)
                photo_json = response.json()
                for photo_data in photo_json:
                    photo_path = self._photo_path(photo_data)
                    self.download_photo(photo_path, photo_data)
                    self._save_json(photo_data)
                    count += 1
                    data_list.append(
                        {
                            'case_id': case_id,
                            'img_id': photo_data['image_id'],
                            'type': photo_data['image_type'],
                            'filename': os.path.basename(photo_path),
                            'path': photo_path,
                            'age_year': photo_data['age_years'],
                            'age_month': photo_data['age_months'],
                            'ethnicity': photo_data['patient_case']['ethnicity'],
                            'gender': photo_data['patient_case']['gender'],
                            'diagnoses': photo_data['patient_case']['diagnoses'],
                            'is_analyzable': photo_data['is_analyzable']
                        }
                    )
            else:
                msg = '%s: code: %s, content: %s' % (case_id, response.status_code, response.content.decode('UTF-8'))
                logging.warning(msg)
                no_photo_list.append(case_id)
            #if count >= 30:
            #    break

        df = pd.DataFrame(data_list)
        df.to_csv(os.path.join(self.project_dir, 'photo_summary.csv'))

    def download_photo(self, file_path, data):
        '''Download photo

        Args:
            file_path -- the path for storing photo
            data -- dict which contains 'image_id' and 'case_id'
        '''
        img_id = data['image_id']
        case_id = data['case_id']
        # https://app.face2gene.com/cases/225703/images/320719/bin?rotate=0&download=true&image_type=no_crop
        url = 'https://app.face2gene.com/cases/%s/images/%s/bin?download=true' % (case_id, img_id)
        photo_result = self.get(url)
        with open(file_path, "wb") as code:
            code.write(photo_result.content)
            logging.info('Download photo - %s_%s', case_id, img_id)

    def _connect(self, user, password):
        payload = {
            'email': user,
            'password': hashlib.md5(password.encode('utf-8')).hexdigest(),
            'userAgent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/71.0.3578.98 Safari/537.36'
            )
        }
        self.post(self.login_url, data=payload).content

    def _save_json(self, data):
        img_id = data['image_id']
        case_id = data['case_id']
        file_dir = os.path.join(self.json_dir, case_id)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filename = case_id + '_' + img_id + '.json'
        file_path = os.path.join(file_dir, filename)
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)

    def _photo_path(self, data):
        img_id = data['image_id']
        name = data['filename']
        case_id = data['case_id']
        file_dir = os.path.join(self.photo_dir, case_id)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filename = case_id + '_' + img_id + '.' + name.split('.')[-1]
        return os.path.join(file_dir, filename)

    def _init_path(self, config):
        project_dir = os.path.join(config['data_path'], config['project_name'])
        photo_dir = os.path.join(project_dir, 'photo')
        json_dir = os.path.join(project_dir, 'json')

        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        if not os.path.exists(photo_dir):
            os.makedirs(photo_dir)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        return project_dir, photo_dir, json_dir
