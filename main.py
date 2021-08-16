# coding=utf-8
import csv
import glob
import os
import re
import numpy as np

import cchardet
import requests
from datahelper import make_csv, readTestCSV
import pandas as pd
import datetime


class TestData:
    def __init__(self):
        self.start_time = datetime.datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.end_time = datetime.datetime.strptime('2018-01-01 23:59:59', '%Y-%m-%d %H:%M:%S')

    def get_s_area_ids(self):
        file_name = '重跑.csv'
        file = open(file_name, 'rb')
        encoding = cchardet.detect(file.read())['encoding']
        csvfile = open(file_name, newline='', encoding=encoding)
        reader = csv.DictReader(csvfile, delimiter=',')
        s_area_id_dic = {}
        for row in reader:
            s_area_id_dic[row.get('頻道ID')] = row.get('頻道名稱')
        return s_area_id_dic

    def get_gender_terms(self):
        gender_term_dic = {"male_terms": [], "female_terms": []}
        male_term_file = 'terms/maleTerm.txt'
        for term in open(male_term_file, encoding='utf-8'):
            gender_term_dic["male_terms"].append(re.sub('\n', '', term))

        female_term_file = 'terms/femaleTerm.txt'
        for term in open(female_term_file, encoding='utf-8'):
            gender_term_dic["female_terms"].append(re.sub('\n', '', term))

        return gender_term_dic

    def by_keyword(self, s_area_id, keyword, limit):
        res = requests.get('http://10.10.10.49:8000/posts?',
                           params={
                               "s_area_id": s_area_id,
                               "keywords": keyword,
                               "limit": limit
                           }).json()
        print(len(res))

    def by_s_area_id(self, s_area_id):
        start_time = self.start_time
        end_time = self.end_time
        res_list = []

        for day in range(1, 639):
            res_list += requests.get('http://10.10.10.49:8000/posts?',
                                     params={
                                         "s_area_id": s_area_id,
                                         "limit": 10,
                                         "start_time": start_time,
                                         "end_time": end_time,
                                         "offset": np.random.randint(0, 50)
                                     }).json()

            start_time += datetime.timedelta(days=1)
            end_time += datetime.timedelta(days=1)

        if len(res_list) <= 100:
            res_list = []
            start_time = self.start_time
            end_time = self.end_time
            for day in range(1, 639):
                res_list += requests.get('http://10.10.10.49:8000/posts?',
                                         params={
                                             "s_area_id": s_area_id,
                                             "limit": 10,
                                             "start_time": start_time,
                                             "end_time": end_time
                                         }).json()

                start_time += datetime.timedelta(days=1)
                end_time += datetime.timedelta(days=1)

        posts = []
        for res in res_list:
            content = res.get('content')
            content = self.clean_content(content)
            if content:
                posts.append(content)

        if len(posts) > 0:
            make_csv({'content': posts, 'label': [None] * len(posts)}, f'test_dcard_data/{s_area_id}.csv')
            print(f'{s_area_id}: {len(posts)}')

        return s_area_id, posts

    def clean_content(self, content):
        url_reg = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        clear_article = re.sub(url_reg, '', content)
        clear_article = re.sub(r'[\s]', '', clear_article)
        return clear_article


def make_test_csv():
    data = TestData()
    s_area_id_dic = data.get_s_area_ids()

    outline_dic = {'s_area_ids': s_area_id_dic.keys(), '頻道': s_area_id_dic.values(), 'posts_num': []}
    test_x = []
    s_area_ids = []
    for id_ in s_area_id_dic.keys():
        s_area_id, posts = data.by_s_area_id(id_)

        if posts:
            test_x += posts
            s_area_ids += [s_area_id] * len(posts)

        outline_dic['posts_num'].append(len(posts))

    # make_csv(outline_dic, 'test_dcard_data/outline.csv')
    # make_csv({'s_area_id': s_area_ids, 'content': test_x, 'label': [None] * len(test_x), 'pac': [None] * len(test_x),'xgboost': [None] * len(test_x)},'test_dcard_data/test.csv')


def predict_csv_file(file_name: str):
    model_types = ['xgboost', 'pac']
    file = open(file_name, 'rb')
    encoding = cchardet.detect(file.read())['encoding']
    csvfile = open(file_name, newline='', encoding=encoding)
    reader = csv.DictReader(csvfile, delimiter=',')

    xgboost_pre = []
    pac_pre = []
    for row in reader:
        content = row.get('content')

        for model_type in model_types:
            res = requests.post(f'http://10.10.10.27:8000/models/{model_type}', json={'content': content}).json()

            if model_type == 'xgboost':
                xgboost_pre.append(res['predict'])

            elif model_type == 'pac':
                pac_pre.append(res['predict'])

    csv_input = pd.read_csv(file_name)
    csv_input['xgboost'] = xgboost_pre
    csv_input['pac'] = pac_pre
    csv_input.to_csv(file_name, index=False)
    print(f'Finish prediction of {file_name}')


# predict_csv_file('test_dcard_data/test.csv')


result = glob.glob('test_dcard_data/**.csv')
# s_area_ids = []
# test_x = []
#
for file_name in result:
    predict_csv_file(file_name)
#     file = open(file_name, 'rb')
#     encoding = cchardet.detect(file.read())['encoding']
#     csvfile = open(file_name, newline='', encoding=encoding)
#     reader = csv.DictReader(csvfile, delimiter=',')
#
#     s_area_id = file_name.split('\\')[-1].split('.')[0]
#     for row in reader:
#         s_area_ids.append(s_area_id)
#         test_x.append(row.get('content'))
#
#
#
# make_csv({'s_area_id':s_area_ids, 'content': test_x, 'label': [None] * len(test_x), 'pac': [None] * len(test_x),'xgboost': [None] * len(test_x)},'test_dcard_data/test.csv')
