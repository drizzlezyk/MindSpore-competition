#定义目标值
TARGET = 'is_depression'

#提交的数据的列拆入预测出来的y_pre数据
data[TARGET] = y_pre
#保存数据命名为test.txt
data.to_csv('test.txt',sep='\n',index=0,header=0)

data