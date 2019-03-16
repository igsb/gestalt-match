#!/usr/bin/env python3

import numpy as np
import csv
import pandas as pd
import os
import shutil
import json

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
    config = configparser.ConfigParser()
    config.read('config.ini')
    working_dir = os.path.join(config['download']['data_path'],
                               config['download']['project_name'],
                               'working/photo')
    sum_file = os.path.join(config['download']['data_path'],
                            config['download']['project_name'],
                            'photo_summary.csv')
    df = load_df(sum_file)
    copy_photo_to_working_dir(df, working_dir)

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
