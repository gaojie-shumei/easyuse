'''
Created on 2019年7月10日

@author: gaojiexcq
'''
from datautil import nlpDataUtil
a = "1234567890"
b = "0987654"
a = [list(a)]
a.append(list(b))
print(a)
print("!"*33)
datautil = nlpDataUtil.NLPDataUtil()
pad_data, pad_y_data, actual_lengths = datautil.padding(a)
print(pad_data)
print(actual_lengths)