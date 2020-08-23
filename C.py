#Step1引用工具包及加载数据
import seaborn as sns
import matplotlib as plt
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

digits =load_digits()
data = digits.data
#print(data.shape) # hape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数。举例说明：建立一个3×3的单位矩阵e, e.shape为（3，3），表示3行3列,第一维的长度为3，第二维的长度也为3
#数据集介绍: 1797个样本，每个样本包括88像素的图像和一个[0, 9]整数的标签。array矩阵类型数据，保存88的图像，里面的元素是float64类型，共有1797张图片 用于显示图片。

#Step2数据预处理：方便后续处理
# 获取第一张图片的像素数
#print(digits.images[0])

# 将XX%的数据作为测试集，其余作为训练集

    #test_size：可以为浮点、整数或None，默认为None

        #①若为浮点时，表示测试集占总样本的百分比

        #②若为整数时，表示测试样本样本数

        #③若为None时，test size自动设置成0.25

    #train_size：可以为浮点、整数或None，默认为None

        #①若为浮点时，表示训练集占总样本的百分比

        #②若为整数时，表示训练样本的样本数

        #③若为None时，train_size自动被设置成0.75

    #random_state：可以为整数、RandomState实例或None，默认为None

        #①若为None时，每次生成的数据都是随机，可能不一样

        #②若为整数时，每次生成的数据都相同

    #stratify：可以为类似数组或None

        #①若为None时，划分出来的测试集或训练集中，其类标签的比例也是随机的

        #②若不为None时，划分出来的测试集或训练集中，其类标签的比例同输入的数组中类标签的比例相同，可以用于处理不均衡的数据集

train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size = 0.1, random_state = 33) 

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

print('-----------------------------------------------------------------------------------------------------------')
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size = 0.3, random_state = None) 

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

print('-----------------------------------------------------------------------------------------------------------')
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size = 0.2, random_state = None) 

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

    #标准化（Z-Score），或者去除均值和方差缩放。公式为：(X-mean)/std  计算时对每个属性/每列分别进行。将数据按期属性（按列进行）减去其均值，并处以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。
    #实现时，有两种不同的方式：
    # 1. 使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。
    # 2. 使用sklearn.preprocessing.scale()函数，可以直接将给定数据进行标准化。

    # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导 ss = StandardScaler()
    # fit_transform()先拟合数据，再标准化， X_train = ss.fit_transform(X_train)。即fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式
    # transform()函数, 即tranform()的作用是通过找中心和缩放等实现标准化
    # 为了数据归一化（使特征数据方差为1，均值为0），我们需要计算特征数据的均值μ和方差σ^2，再使用下面的公式进行归一化：
    # 我们在训练集上调用fit_transform()，其实找到了均值μ和方差σ^2，即我们已经找到了转换规则，我们把这个规则利用在训练集上，同样，我们可以直接将其运用到测试集上（甚至交叉验证集），所以在测试集上的处理，我们只需要标准化数据而不需要再次拟合数据。


#Step3选择模型，比如LR,CART决策树
    #CART 算法简单介绍 Classification And Regression Tree，即分类回归树算法，简称CART算法，它是决策树的一种实现，通常决策树主要有三种实现，分别是ID3算法，CART算法和C4.5算法。CART 算法采用 Gini系数作为标准进行特征分割。

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=88, splitter = 'best', criterion='gini')
clf.fit(train_ss_x, train_y)


#Step4训练模型（训练集）
#训练一个DecisionTree分类器

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                        max_features=None, max_leaf_nodes=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, presort=False,
                        random_state=0, splitter='best')


#Step5模型评估（测试集）
predit_y = clf.predict(test_ss_x)
print('CARTs  accurancy: %lf'% accuracy_score(test_y, predit_y))
