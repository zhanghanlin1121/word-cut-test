import gensim
import codecs
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from gensim.models import word2vec
from gensim.models import KeyedVectors

# ##读取文件
# sentences=word2vec.Text8Corpus('training.txt')
# ##训练
# model=word2vec.Word2Vec(sentences, size=50,sample=0,min_count=0,window = 2)
# #保存
# model.wv.save_word2vec_format("vec.txt")
def listx(num):
    lis = []
    for j in range(num): lis.append(j)
    return lis
def load_feature_words(name):
    data_set_fp = xlrd.open_workbook("word_set.xlsx")
    table1 = data_set_fp.sheet_by_name(name)
    word_vectors = listx(table1.nrows)
    for i in range(0, table1.nrows):
        word_vectors[i] = table1.row_values(i)[1]
    return word_vectors

word_vectors = KeyedVectors.load_word2vec_format('vec.txt', binary=False)

visualizeWords =load_feature_words(u'主业务标签')

# features = ['白条']
# targetFeatures = ['bbbbbb']
# targetFeatures = ['人脸识别', '申请']
#查看相似度 可以用n_most_similar 来查看多个序列组合的相似度
print(word_vectors.wv.similar_by_word("白条"))
print(word_vectors.wv.similar_by_word("小金库"))
print(word_vectors["白条"])
#
# visualizeWords = [
#     "白条", "小金库","小白卡","金条","分期","免息"]
tokens1 = {}

# visualizeIdx =
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
visualizeVecs = []
# try:
visualizeVecs = [word_vectors[word] for word in visualizeWords]
# except:
#     print("发生异常")
# visualizeVecs = word_vectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeWords) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')

# model.n_similarity()
# trainingFile = open("training.txt",encoding='utf-8')
# outputFile = open('sentence.txt', 'w',encoding='utf-8')
# index = 0
# while 1:
#     line = trainingFile.readline()
#     if not line:
#         break
#     if line[-1] == "\n":
#         line = line[0:-1]
#     labelList = line.split(sep="\t")
#     result = np.zeros((50,), dtype=np.float32)
#     for label in labelList:
#         if label is None or label == "":
#             continue
#         vec = model.wv[label]
#         assert vec.shape == (50,)
#         result += vec
#     index += 1
#     outputFile.writelines(str(index) + '\t' + str(result) + '\n')
#     if index % 100 == 0:
#         outputFile.flush()
#     if index == 5000:
#         break
# print(y2)
#
