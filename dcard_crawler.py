import requests
from bs4 import BeautifulSoup
from tensorflow import keras

from datahelper import MakeCSV,readTestCSV,readTrainCSV,appendData
from modelHelper import AlbertModel

class DcardCrawler:
    def __init__(self):
        self.base_url = 'https://www.dcard.tw/f/'
        self.min_word = 20

    def crawler(self,topic):
        titles = []
        contents = []
        labels = []
        url = self.base_url + topic
        res = requests.get(url).text
        soup = BeautifulSoup(res, 'html.parser')
        for post in soup.find_all('article'):
            title = post.find('span').text
            gender = post.find('svg').find('title')
            if gender:
                content_url = post.find('a').get('href')
                res = requests.get('https://www.dcard.tw' + content_url).text
                soup = BeautifulSoup(res, 'html.parser')
                content = ''
                for article in soup.find_all('article'):
                    for para in article.find_all('span'):
                        content += para.text.replace('\n','')

                if len(content) > self.min_word:
                    titles.append(title)
                    contents.append(content)
                    if gender.text == '男':
                        labels.append(1)
                    else:
                        labels.append(0)
        return titles,contents,labels

dcard = DcardCrawler()
# topics = ['stayhome','2019_ncov','makeup','graduate','freshman','stock','buyonline','delivery','house','cooking',
          # 'fitness','nba','netflix','relationship','mood','talk','game','basketball','esports']
# contents = []
# labels = []
# topics = ['sneakers','soccer','baseball','basketball','esports']


# for topic in topics:
#     title,content,label = dcard.crawler(topic)
#     print(f'{topic} : {len(content)}筆')
#     contents += content
#     labels += label
# #
# print('--------------------------------------------')
# print(f'total dcrad crawler data : {len(contents)}筆')
# MakeCSV(contents,labels,'dcard_crawler.csv')

contents,labels = readTestCSV('train_data.csv')
for index,c in enumerate(contents):
    if '婆婆' in c:
        print(c)
        print(labels[index])
# print(len(contents))

# right = 0

# albert = AlbertModel()
# result = albert.predict(['如題，剛剛在吃飯的時候，阿嬤突然說那個住隔壁鄰居真好命，生兩個兒子，然後又跟我婆婆說，你也很好命耶！你有兩個兒子！此時的我沉默，因為我前年生了一個女兒，最近又被暗示要再拼一個兒子。'])
# print(result)

# false_contents = []
# false_labels = []

# for i in range(len(result)):
#     pre = round(result[i][0])
#     if pre != labels[i]:
#         print(contents[i])
#         false_contents.append(contents[i])
#         print(result[i][0])
#         print(labels[i])
#         false_labels.append(labels[i])
#
#     else:
#         right += 1
# print(f'Accuracy : {right / len(result)}')

# albert.modify_model(false_contents,false_labels)


