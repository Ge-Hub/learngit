import time
import pandas as pd

dataset = pd.read_csv('MBO.csv', header = None)
print(dataset.head())
print('-'*80)
print(dataset.shape)

transactions = []

for i in range (0, dataset.shape[0]):
    temp = []
    for j in range (0, dataset.shape[1]):
        if str(dataset.values[i,j] != 'nan'):
            temp.append (str(dataset.values[i,j]))
    #print(temp)

transactions.append(temp)
#print(transactions)
print('-'*88)

# Rule1 efficient_apriori：
print('-'*50, 'efficient_apriori', '-'*50)

def rule1():
    from efficient_apriori import apriori
    start = time.time()
    
    itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.5)
    print('频繁项集：', itemsets)
    print('-'*50)
    print('关联规则：', rules)
    end = time.time()
    print("用时：", end-start)

    end = time.time()
    print("用时：", end-start)

rule1()


