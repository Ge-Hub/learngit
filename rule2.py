#步骤1：导入库 
import time
import pandas as pd

#步骤2：载入资料集并查看

dataset = pd.read_csv('MBO.csv', sep="\t", header = None)
dataset2 = dataset[0].str.get_dummies(sep=",")
print(dataset2.head())
print('-'*80)
print(dataset2.shape)

# Rule2 mixtend.frequent_patterns：
print('-'*50, 'mixtend.frequent_patterns', '-'*50)

def rule2():
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    start = time.time()

    
    # 按照支持度从大到小进行时候粗
    itemsets = apriori(dataset2, use_colnames=True, min_support=0.02)
    itemsets = itemsets.sort_values(by="support" , ascending=False) 
    print('-'*20, '频繁项集', '-'*20)
    print(itemsets)
    
    # 根据频繁项集计算关联规则，设置最小提升度为2
    rules = association_rules(itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values(by="lift" , ascending=False)
    print('-'*20, '关联规则', '-'*20)
    print(rules)
    rules.to_csv('./rules.csv')

    end = time.time()
    print("用时：", end-start)

rule2()

