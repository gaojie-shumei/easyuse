'''
Created on 2019年7月19日

@author: gaojiexcq
'''
import os
import os.path as ospath
import pandas as pd
import numpy as np

def get_all_file(base_dir,filenamelike=None,filetype=None,uncased=False):
    '''
    :base_dir: the root dir
    :filenamelike: a file name is readme-01.txt  the filenamelike is for readme-01,the txt is the filetype
    :filetype: the file type,such as txt,doc,docx,pdf
    :uncased: if True,('read'=='Read') is True,if False,('read'=='Read') is False
    '''
    flist = os.listdir(base_dir)
    filepathlist = []
    for filename in flist:
        if ospath.isdir(ospath.join(base_dir,filename)):
            filepathlist = filepathlist + get_all_file(ospath.join(base_dir,filename),filenamelike,filetype,uncased)
        else:
            splitname= filename.rsplit(sep=".", maxsplit=1)
            if uncased:
                splitname[0] = splitname[0].lower()
                splitname[1] = splitname[1].lower()
                filenamelike = filenamelike.lower()
                filetype = filetype.lower()
            if filenamelike is not None and filenamelike in splitname[0].lower():
                if filetype is not None and filetype in splitname[1].lower():
                    filepathlist.append(ospath.join(base_dir,filename))
                if filetype is None:
                    filepathlist.append(ospath.join(base_dir,filename))
            if filenamelike is None:
                if filetype is not None and filetype in splitname[1].lower():
                    filepathlist.append(ospath.join(base_dir,filename))
                if filetype is None:
                    filepathlist.append(ospath.join(base_dir,filename))
    sorted(filepathlist)
    return filepathlist

def onebyone(readme_path_list,classfication_path_list,labeling_path_list,split_pattern=" "):
    info_list = []
    for readme_path in readme_path_list:
        info_dict = {"readme_path":readme_path}
        readme_name = ospath.basename(readme_path).rsplit(".",1)[0]
        readme_dir = ospath.dirname(readme_path)
        for classfication_path in classfication_path_list:
            classfication_name = ospath.basename(classfication_path).rsplit(".",1)[0]
            classfication_dir = ospath.dirname(classfication_path)
            if readme_name.split(split_pattern)[-1]==classfication_name.split(split_pattern)[-1] and readme_dir==classfication_dir:
                info_dict["classification_path"] = classfication_path
                break
        for labeling_path in labeling_path_list:
            labeling_name = ospath.basename(labeling_path).rsplit(".",1)[0]
            labeling_dir = ospath.dirname(labeling_path)
            if readme_name.split(split_pattern)[-1]==labeling_name.split(split_pattern)[-1] and readme_dir==labeling_dir:
                info_dict["labeling_path"] = labeling_path
                break
        info_list.append(info_dict)
    return info_list

def extract_data(info_list,key_value_split_pattern=":",value_split_pattern=",",encoding="utf-8"):
    for i in range(len(info_list)):
        info_dict = info_list[i]
        with open(info_dict["readme_path"],mode="r+",encoding=encoding) as f:
            try:
                for line in f:
                    line = line.strip().replace("\ufeff","")
                    if key_value_split_pattern in line:
                        line_list = line.rsplit(key_value_split_pattern,1)
                        if len(line_list)==1:
                            info_dict[line_list[0].strip().lower()] = []
                        else:
                            if line_list[1].strip()=="":
                                info_dict[line_list[0].strip().lower()] = []
                            else:
                                value_list = line_list[1].split(value_split_pattern)
                                value_list = [int(val.strip()) for val in value_list]
                                info_dict[line_list[0].strip().lower()] = value_list
            except:
                print("file {} is not utf-8".format(info_dict["readme_path"]))
                # raise(RuntimeError("file encode not is utf-8"))
        info_list[i] = info_dict
    return info_list

"""拼接一个文件中所有json文件为一个dataframe"""
def restore_data(info_list,csvpath): #参数为30行的日期，注意修改自己的文件夹路径
    data = pd.DataFrame(columns=['annotation_approver','annotations','id','meta','text'])#相当于初始化
    all_data = pd.DataFrame(columns=['id', 'text', 'label'])#相当于初始化
    df = []
    for info in info_list:
        try:
            df = pd.read_json(info["classification_path"],orient='records',encoding='utf-8',lines=True)
        except KeyError as ex:
            print(ex,"     ",info["readme_path"])
        new_data = pd.concat([data, df], axis=0, ignore_index=True)
        """提取id,label,text三列"""
        new_data['label'] = new_data[new_data['annotations'].map(lambda x: len(x) > 0)]['annotations'].map(lambda x: x[0]['label'])
        columns = ['id', 'text', 'label']
        new_data = new_data[columns]
        new_data = new_data.fillna(0)  # 需要重新赋值数据才能更改
        label_to = {}
        try:
            for not_negative in info["not negative"]:
                label_to[not_negative] = 1
            for negative_but_not_material in info["negative but not material"]:
                label_to[negative_but_not_material] = 2
            for negetive_and_material in info["negative and material"]:
                label_to[negetive_and_material] = 3
        except KeyError as e:
            print(e,"    ",info)
#         print(info["not negative"],info["negative but not material"],info["negetive and material"],label_to)
        new_data[['label']] = new_data[['label']].astype(int)  # 恢复label为int型
        new_data.replace({'label':label_to},inplace=True)
        all_data = pd.concat([all_data,new_data],axis=0,ignore_index=True)
    all_data.to_csv(csvpath,encoding='utf-8',index=True,header='id,text,label')
    return all_data#返回拼接好的dataframe


def main():
    #base_dir = "C:/Users/gaojiexcq/Desktop/数据处理/2019-7-12"
    base_dir = "E:/text_cla/dataset"
    readme_path_list = get_all_file(base_dir, "readme", "txt", True)
    classfication_path_list = get_all_file(base_dir, "classification", "json", True)
    labeling_path_list = get_all_file(base_dir, "labeling", "json", True)
    info_list = onebyone(readme_path_list, classfication_path_list, labeling_path_list, split_pattern=" ")
    info_list = extract_data(info_list, key_value_split_pattern=":", value_split_pattern=",")
    # print(info_list)
    restore_data(info_list,"sub.csv")
    for info in info_list:
        print(info)

if __name__ == '__main__':
    main()    