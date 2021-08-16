import csv
from pathlib import Path
import cchardet
from sklearn.utils import shuffle
import re


class GetTrainData:
    def __init__(self):
        self.root = Path.cwd()

    def getMaleTerms(self):
        path = 'terms/maleTerm.txt'
        file = open(path)
        maleTerms = []
        for term in file:
            maleTerms.append(term.replace('\n',''))
        return maleTerms

    def getFemaleTerm(self):
        path = 'terms/femaleTerm.txt'
        file = open(path)
        femaleTerms = []
        for term in file:
            femaleTerms.append(term.replace('\n',''))
        return femaleTerms

    def getMaleData(self,name):
        contents_dcard_male = self.readDcardCSV(f'{name}.csv','M')
        return contents_dcard_male

    def getFemaleData(self,name):
        contents_dcard_female = self.readDcardCSV(f'{name}.csv', 'F')
        return contents_dcard_female

    def readPTT(self,name):
        file_name = name + '.csv'
        file = open(file_name, 'rb')
        encoding = cchardet.detect(file.read())['encoding']

        csvfile = open(file_name, newline='', encoding=encoding)
        reader = csv.DictReader(csvfile, delimiter='\t')

        male_terms = self.getMaleTerms()
        female_terms = self.getFemaleTerm()

        contents = []
        labels = []


        for row in reader:
            content = row.get('content')
            if content and len(contents) <= 512:
                for term in male_terms:
                    if term in content:
                        contents.append(content)
                        labels.append(1)
                        break
        csvfile.seek(0)


        for row in reader:
            content = row.get('content')
            if content and len(contents) <= 512:
                for term in female_terms:
                    if term in content:
                        contents.append(content)
                        labels.append(0)
                        break

        return contents,labels

    def readDcardCSV(self,file_name,chosen_gender):
        csv.field_size_limit(500 * 1024 * 1024)
        male_terms = self.getMaleTerms()
        file = open(file_name, 'rb')
        encoding = cchardet.detect(file.read())['encoding']

        delimiter = '\t'
        if file_name == 'dcard/dcard_2021_02_n50000_t100.csv':
            delimiter = ','

        csvfile = open(file_name, newline='', encoding=encoding)
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        contents = []

        if chosen_gender == 'M':
            for row in reader:
                content = row.get('content')
                if content != None:
                    try:
                        content = content.replace('\n', '')
                        gender = row['author'].split('/')[1]
                    except:
                        continue

                    if gender == chosen_gender and len(contents) < 512:
                        for term in male_terms:
                            if term in content:
                                contents.append(content)
                                break

        if chosen_gender == 'F':
            female_terms = self.getFemaleTerm()
            for row in reader:
                content = row.get('content')
                if content != None:
                    try:
                        content = content.replace('\n', '')
                        gender = row['author'].split('/')[1]
                    except:
                        continue

                    if gender == chosen_gender and len(contents) < 512:
                        for term in female_terms:
                            if term in content:
                                contents.append(content)
                                break
        return contents

    def readLabelData(self,file_name):
        file = open(file_name, 'rb')
        encoding = cchardet.detect(file.read())['encoding']

        csvfile = open(file_name, newline='', encoding=encoding)
        reader = csv.DictReader(csvfile, delimiter=',')

        male_content = []
        other_content = []

        male = 0
        female = 0
        for row in reader:
            gender = row['label']
            content = row['content'].replace('\n','').replace(' ','')
            if gender == '1.0':
                male_content.append(content)
                male += 1
            elif gender == '0.0':
                other_content.append(content)
                female += 1

        print(f'male : {male}')
        print(f'female : {female}')

        male_labels = [1] * len(male_content)
        other_labels = [0] * len(other_content)

        contents = male_content + other_content
        labels = male_labels + other_labels

        return contents,labels

def remove_url(article):
    url_reg = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    clear_article = re.sub(url_reg,'',article)
    return clear_article

def channel_to_id(id):
    file_name = 's_area_id_dcard_orig.csv'
    file = open(file_name, 'rb')
    encoding = cchardet.detect(file.read())['encoding']
    csvfile = open(file_name, newline='', encoding=encoding)
    reader = csv.DictReader(csvfile, delimiter=',')

    channel_dic = {}
    for index, row in enumerate(reader):
        channel = row.get('頻道ID')
        channel_dic[channel] = index + 1

    return channel_dic.get(id, 0)

def readTrainCSV(file_name):
    print(file_name)
    file = open(file_name, 'rb')
    encoding = cchardet.detect(file.read())['encoding']
    csvfile = open(file_name, newline='', encoding=encoding,errors='ignore')
    reader = csv.DictReader(x.replace('\0', '') for x in csvfile)
    # reader = csv.DictReader(csvfile, delimiter=',')

    male_channels = []
    male_authors = []
    male_data = []
    male_labels = []

    other_channels = []
    other_authors = []
    other_data = []
    other_labels = []

    for row in reader:
        print(row)
        if row != None:
          # channel = channel_to_id(row.get('channel'))
          # author = row.get('author')
          label = int(float(row.get('label')))
          content = row.get('content')
          content = remove_url(content)
          if label == 1:
              # male_channels.append(channel)
              # male_authors.append(author)
              male_data.append(content)
              male_labels.append(label)
          if label == 0:
              # other_channels.append(channel)
              # other_authors.append(author)
              other_data.append(content)
              other_labels.append(label)

    print(f'male data in csv : {len(male_data)}')
    print(f'other data in csv : {len(other_data)}')
    male_data, male_labels = shuffle(male_data,male_labels)
    other_data, other_labels = shuffle(other_data,other_labels)
    # male_channels,male_authors,male_data,male_labels = shuffle(male_channels,male_authors,male_data,male_labels)
    # other_channels,other_authors,other_data,other_labels = shuffle(other_channels,other_authors,other_data,other_labels)
    use_num = min(len(male_data),len(other_data))

    # channels = male_channels[:use_num] + other_channels[:use_num]
    # authors = male_authors[:use_num] + other_authors[:use_num]
    contents = male_data[:use_num] + other_data[:use_num]
    labels = male_labels[:use_num] + other_labels[:use_num]

    return contents,labels

def readTestCSV(file_name):
    file = open(file_name, 'rb')
    encoding = cchardet.detect(file.read())['encoding']
    csvfile = open(file_name, newline='', encoding=encoding)
    reader = csv.DictReader(x.replace('\0', '') for x in csvfile)
    # reader = csv.DictReader(csvfile, delimiter=',')

    contents = []
    labels = []
    for index,row in enumerate(reader):
        # print(row)
        label = int(float(row.get('label')))
        content = row.get('content')
        contents.append(content)
        labels.append(label)

    contents,labels = shuffle(contents,labels)
    return contents,labels


def make_csv(contents_dic: dict,file_name: str):
    with open(file_name, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(contents_dic.keys())
        writer.writerows(zip(*contents_dic.values()))


def appendData(contents,labels,file_name):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(contents)):
            writer.writerow([contents[i],labels[i]])

def evaluateData(labels):
    male_data = 0
    other_data = 0

    for i in range(len(labels)):
        if labels[i] == 1:
            male_data += 1
        if labels[i] == 0:
            other_data += 1

    print(f'male data : {male_data}')
    print(f'female data : {other_data}')
    print(f'total : {male_data + other_data}')




