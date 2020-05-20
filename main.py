import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba.analyse
import os

stop_words = ['我們', '一個', '一定', '不是', '就是', '不會', '因為', '必須', '這些', '將會', '這是',
              '我會', '這個', '謝謝', '一起', '大家', '將會', '已經', '我要', '可以', '以及', '事情', ]

jieba.suggest_freq('年輕人', True)
jieba.suggest_freq('兩千三百萬', True)


def parse_speech(file):
    token = []
    with open(file, "r") as f:
        content = f.read()

    paragraph = jieba.cut(content, cut_all=False)
    paragraph = [word for word in paragraph if word not in stop_words]
    token.append(" ".join(paragraph))

    return token


def tfidf_anayze(token):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    try:
        tfidf = transformer.fit_transform(vectorizer.fit_transform(token))
        word = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        stats = {}
        for i in range(len(weight)):
            for j in range(len(word)):
                if weight[i][j] > 0.05:
                    # feq.append("#" + str(word[j]))
                    # print(str(word[j] + "," + str(weight[i][j])))
                    stats.update({word[j]: weight[i][j]})


    except:
        print("extract the keywords error")

    return stats


def main():
    file_list = os.listdir('data')
    for file in file_list:
        token = parse_speech(f'data/{file}')
        stats = tfidf_anayze(token)
        stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True)}

        with open(f'sorted_freq/{file}.csv', 'w') as file:
            for k, v in stats.items():
                print(f'{k},{v}', file=file)

            print(f'sorted_freq/{file}.csv complete')

