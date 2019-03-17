#!/usr/bin/env python2
import argparse
import numpy as np
import csv
import pandas as pd
import os
import shutil
import json
import sys
sys.path.append('../insightface/deploy')
import face_model
import cv2
import configparser
import logging

def genomic_data_genes(data):
    genes = []
    classification = []
    if 'genomic_entries' in data['case_data']:
        for g_data in data['case_data']['genomic_entries']:
            # ['TARGETED_TESTING',
            #  'CHROMOSOMAL_MICROARRAY',
            #  'SINGLE_GENE_SEQUENCING',
            #  'EXOME_SEQUENCING',
            #  'FISH',
            #  'KARYOTYPE',
            #  'MULTIGENE_PANEL',
            #  'WHOLE_GENE_SEQUENCING',
            #  'OTHER',
            #  'METHYLATION_TESTING',
            #   None]
            interpretation = []

            test_type = g_data['test_type']
            if (test_type == 'SINGLE_GENE_SEQUENCING' or
                test_type == 'TARGETED_TESTING'):
                if g_data['gene'] != '':
                    genes.append(g_data['gene']['gene_symbol'])
                for v in g_data['variants']:
                    if 'zygosity' not in v:
                        continue
                    if v["zygosity"] == "COMPOUND_HETEROZYGOUS":
                        if 'mutation1' in v and 'interpretation' in v['mutation1']:
                            interpretation.append(v['mutation1']['interpretation'])
                        if 'mutation2' in v and 'interpretation' in v['mutation2']:
                            interpretation.append(v['mutation2']['interpretation'])
                    else:
                        if 'mutation' in v and 'interpretation' in v['mutation']:
                            interpretation.append(v['mutation']['interpretation'])

            elif (test_type == 'EXOME_SEQUENCING' or
                  test_type == 'MULTIGENE_PANEL' or
                  test_type == 'WHOLE_GENE_SEQUENCING'):
                for v in g_data['variants']:
                    if 'gene' in v:
                        genes.append(v['gene']['gene_symbol'])
                    if 'zygosity' not in v:
                        continue
                    if v["zygosity"] == "COMPOUND_HETEROZYGOUS":
                        if 'mutation1' in v and 'interpretation' in v['mutation1']:
                            interpretation.append(v['mutation1']['interpretation'])
                        if 'mutation2' in v and 'interpretation' in v['mutation2']:
                            interpretation.append(v['mutation2']['interpretation'])
                    else:
                        if 'mutation' in v and 'interpretation' in v['mutation']:
                            interpretation.append(v['mutation']['interpretation'])
            elif test_type == 'FISH':
                pass
            elif test_type == 'CHROMOSOMAL_MICROARRAY':
                pass
            elif test_type == 'KARYOTYPE':
                pass
            elif test_type == 'OTHER':
                pass
            elif test_type == 'METHYLATION_TESTING':
                pass
            else:
                pass
            if 'BENIGN' in interpretation and len(interpretation) == 1:
                print(json.dumps(g_data, indent=4, sort_keys=True))
                classification.append('benign')
            else:
                classification.append('pathogenic')
    return genes, classification

def check_diag(x):
    return 'MOLECULARLY_DIAGNOSED' in x['diagnosis'] or 'CLINICALLY_DIAGNOSED' in x['diagnosis']
def filter_benign(x):
    return 'benign' not in x['classification']
def mono_gene(x):
    return len(x['corrected_genes']) == 1
def frontal(x):
    return x['type'] == 'frontalFace'
def check_syndrome(x):
    return len(x['diagnoses']) > 0
def is_analyzable(x):
    return x['is_analyzable']
def check(x):
    return (check_diag(x) and
            filter_benign(x) and
            mono_gene(x) and
            frontal(x) and
            is_analyzable(x) and
            check_syndrome(x)
           )

def load_df(summary_file):
    df_input = pd.read_csv(summary_file)
    #df.is_copy = None
    #df = df.reset_index(drop=True)
    df_input['selected_syndromes'] = None
    df_input['selected_genes'] = None
    df_input['app_valid'] = None
    df_input['diagnosis'] = None
    df_input['genomic_genes'] = None
    df_input['classification'] = None
    df_input['corrected_genes'] = None
    testing = []
    for i in range(df_input.shape[0]):
        case_id = df_input.loc[i]['case_id']#'225703'#'230381'
        json_path = "pedia/%s.json" % case_id
        with open(json_path, 'r') as f:
            data = json.load(f)
        df_input.at[i, 'selected_genes'] = [g['gene_name']for g in data['case_data']['selected_genes']]
        df_input.at[i, 'selected_syndromes'] = [s['syndrome']['syndrome_name'] for s in data['case_data']['selected_syndromes']]
        df_input.at[i, 'diagnosis'] = [d['diagnosis'] for d in data['case_data']['selected_syndromes']]
        df_input.at[i, 'app_valid'] = [s['syndrome']['app_valid'] for s in data['case_data']['selected_syndromes']]
        df_input.at[i, 'genomic_genes'], df_input.at[i, 'classification'] = genomic_data_genes(data)
        df_input.at[i, 'corrected_genes'] = list(set(df_input.at[i, 'selected_genes'] + df_input.at[i, 'genomic_genes']))


    df = df_input.loc[df_input.apply(check, axis=1)]
    return df
def copy_photo_to_working_dir(df, working_dir):
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    for p in df.path:
        name = os.path.basename(p).split('.')[0]
        p_dir = os.path.join(working_dir, name)
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)
        shutil.copy2(p, os.path.join(os.path.join(p_dir, name+'.jpg')))

        
def main():
    '''
    To use arcface to fetch the embeddings, please run the following commad
    python2.7 arc_face.py --model ../insightface/models/model-r100-ii/model,0000
    --ga-model ../insightface/models/model-r100-ii/model,0000
    '''
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()

    logging.basicConfig(
        filename=os.path.join('log/arcface.log'),
        filemode='w',
        format='%(asctime)s: %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    config = configparser.ConfigParser()
    config.read('config.ini')
    working_dir = os.path.join(config['Download']['data_path'],
                               config['Download']['project_name'],
                               'working/align_112')
    sum_file = os.path.join(config['Download']['data_path'],
                            config['Download']['project_name'],
                            'photo_summary.csv')
    arc_dir = os.path.join(config['Download']['data_path'],
                          config['Download']['project_name'],
                          'working/arcface')
    if not os.path.exists(arc_dir):
        os.makedirs(arc_dir)
    emb_dir = os.path.join(arc_dir, 'embeddings')
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)

    df = load_df(sum_file)
    model = face_model.FaceModel(args)
    count = 0
    feature = np.array([]).reshape(0, 512)
    label = np.array([]).reshape(0, 1)
    err_count = 0
    for pic_path in os.listdir(working_dir):
        print(pic_path)
        pic_dir = os.path.join(working_dir, pic_path)
        if not os.path.isdir(pic_dir):
            continue
        if len(os.listdir(pic_dir)) == 0:
            continue
        name = os.listdir(pic_dir)[0]
        pic_file = os.path.join(pic_dir, name)
        img = cv2.imread(pic_file)
        #img = cv2.imread('../insightface/deploy/Tom_Hanks_54745.png')
        #img = cv2.imread('data/pedia/working/align_112/101329_136767/101329_136767.png')

        img = model.get_input(img)
        try: 
            f1 = model.get_feature(img)
        except Exception:
            logging.error('Arcface can not process photo: ' + pic_path)
            err_count += 1
            continue
        np.save(os.path.join(emb_dir, pic_path + '.npy'), f1)
        feature = np.vstack([feature, [f1]])
        label = np.append(label, pic_path.encode("utf-8"))
        count += 1

    np.save(os.path.join(arc_dir, 'label.npy'), label)
    np.save(os.path.join(arc_dir, 'embeddings.npy'), feature)
    print('Arcface processed %d photos sucessfully.' % count)
    print('Arcface can not processed %d photos sucessfully.' % err_count)
    print(feature.shape)
    print(label)


if __name__ == '__main__':
    main()

#facenet_path = '../facenet/contributed/export_embeddings.py'
#
#cmd = 'python ' + facenet_path + ''


# 1. Read the parameter including
#  - input folder
#  - output folder
#  - config parameter
# 2. Some file name preprocessing
#  - get case ID:wq

#  - remove photo which is not frontal photo
# 3. Preocess facenet to export the embeddings
# 4. Output
#  - output summary file
#  - output all embeddings
