import pandas as pd

from sklearn.model_selection import train_test_split

#读取当前目录下的数据
data = pd.read_csv('./depression_dataset_reddit_cleaned.csv')

#将数据拆分成比赛使用的和后面评分使用的数据集，保存为train和test
train,test = train_test_split(data,test_size=0.5,random_state=1024)

#保存数据到制定路径
train.to_csv('./data/train.csv')
test.to_csv('./data/test.csv')


#读取比赛使用的train数据
train = pd.read_csv('./data/train.csv')

#拆分特征和目标值
X = train[['clean_text']]
y = train[['is_depression']]

#拆分数据,将目标值和特征拆分为X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1024)