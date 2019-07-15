'''
Created on 2019年7月12日

@author: gaojiexcq
'''
'''
读取数据
data,data_pos_tag,data_chunk_tag是x数据
label是y数据
label_index 是所有的类别，包括实体和非实体
'''
def read_data(filepath,encoding="utf-8",position=0,read_data_size=None,padding_str="<pad>"):
    data = []
    data_pos_tag = []
    data_chunk_tag = []
    label = []
    label_index = []
    
    sub_data = []
    sub_data_pos_tag = []
    sub_data_chunk_tag = []
    sub_label = []
    with open(filepath,mode="r+",encoding=encoding) as f:
        f.seek(position)
        count = 0
        while True:
            line = f.readline()
            if line=="":
                break
            line = line.strip()
            if position!=0 and line.strip()!="":
                line_split = line.split(" ")
                sub_data.append(line_split[0])
                sub_data_pos_tag.append(line_split[1])
                sub_data_chunk_tag.append(line_split[2])
                sub_label.append(line_split[3])
                label_index.append(line_split[3])
            elif len(sub_data)>0:
                data.append(sub_data)
                data_pos_tag.append(sub_data_pos_tag)
                data_chunk_tag.append(sub_data_chunk_tag)
                label.append(sub_label)
                sub_data = []
                sub_data_pos_tag = []
                sub_data_chunk_tag = []
                sub_label = []
                count+=1
            position = 1
            if read_data_size!=None and count==read_data_size:
                break
        position = f.tell()
        label_index = list(set(label_index))
        label_index.sort()
        label_index = [padding_str]+label_index
    return data,data_pos_tag,data_chunk_tag,label,position,label_index