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
        if ospath.isdir(ospath.join(base_dir, filename)):
            filepathlist = filepathlist + get_all_file(ospath.join(base_dir, filename), filenamelike, filetype, uncased)
        else:
            splitname= filename.rsplit(sep=".", maxsplit=1)
            if uncased:
                try:
                    splitname[0] = splitname[0].lower()
                    if len(splitname)>1:
                        splitname[1] = splitname[1].lower()
                    else:
                        filetype = None
                except:
                    print("Error  :"+ospath.join(base_dir,filename))
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


def classification_data_info_store(info_list, jsonpath, processjson_info_path, depreated_text="DirtyDeedsDoneDirtCheap"):
    # data = pd.DataFrame(columns=['annotation_approver','annotations','id','meta','text'])
    all_data = pd.DataFrame(columns=['id', 'text', 'label'])
    for info in info_list:
        try:
            df = pd.read_json(info["classification_path"], orient='records', encoding='utf-8', lines=True)
        except KeyError as ex:
            print("Error:", ex, "     ", info["readme_path"])
        # new_data = pd.concat([data, df], axis=0, ignore_index=True)
        """提取id,label,text三列"""
        # try:
        columns = ['id', 'text', 'label']
        dfdata = np.array(df[["id", "text", "annotations"]])
        dflabel = []
        for i in range(dfdata.shape[0]):
            try:
                if len(dfdata[i,2])>0:
                    dflabel.append(dfdata[i, 2][0]["label"])
                else:
                    dflabel.append(-1)
            except:
                print("Error:id=", dfdata[i, 0], ",info=", info)
                dflabel.append(-1)
        dflabel = np.array(dflabel)
        df = pd.DataFrame(np.c_[dfdata[:, 0:2], dflabel], columns=columns)
        # df['label'] = df[df['annotations'].map(lambda x: len(x) > 0)]['annotations'].map(lambda x: x[0]['label'])
        # except:
        #     print("Error:", info)

        # df = df[columns]
        df["label"] = df["label"].fillna(-1)  # 需要重新赋值数据才能更改
        label_to = {}
        try:
            for not_negative in info["not negative"]:
                label_to[not_negative] = 0
            for negative_but_not_material in info["negative but not material"]:
                label_to[negative_but_not_material] = 1
            for negetive_and_material in info["negative and material"]:
                label_to[negetive_and_material] = 2
        except KeyError as e:
            print("Error:", e, "    ", info)
#         print(info["not negative"],info["negative but not material"],info["negetive and material"],label_to)
#         print(df.columns)
        df['label'] = df['label'].astype(np.int32)  # 恢复label为int型
        df['label'].replace(label_to, inplace=True)
        all_data = pd.concat([all_data, df], axis=0)
    all_data.to_json(jsonpath, orient="records", lines=True)
    if depreated_text is not None and depreated_text != "":
        all_data["text"].replace(depreated_text + "(:|：)*([0-9]*)", "", regex=True, inplace=True)
    data = np.array(all_data)
    data = np.delete(data, np.where(data[:, 1] == "")[0].reshape(-1), axis=0)
    delete_index = []
    for i in range(data.shape[0]):
        if data[i,2] not in [-1, 0, 1, 2]:
            delete_index.append(i)
    data = np.delete(data, delete_index, axis=0)
    labels = np.unique(data[:,2])
    processjson_info_str = "labels:"+str(labels)+"\n"
    for lb in labels:
        percent = np.sum(data[:, 2] == lb)/data.shape[0]
        if lb == -1:
            processjson_info_str += "not sure:{:f} \n".format(percent)
        elif lb==0:
            processjson_info_str += "not negative:{:f} \n".format(percent)
        elif lb==1:
            processjson_info_str += "negative but not material:{:f} \n".format(percent)
        elif lb==2:
            processjson_info_str += "negative and material:{:f} \n".format(percent)
        else:
            processjson_info_str += "error:{:f} \n".format(percent)
    with open(processjson_info_path,mode="w",encoding="utf-8") as f:
        f.write(processjson_info_str)
    return all_data


def main():
    #base_dir = "C:/Users/gaojiexcq/Desktop/数据处理/2019-7-12"
    # base_dir = "E:/text_cla/dataset"
    base_dir = "D:/数据/dataset"
    readme_path_list = get_all_file(base_dir, "readme", "txt", True)
    classfication_path_list = get_all_file(base_dir, "classification", "json", True)
    labeling_path_list = get_all_file(base_dir, "labeling", "json", True)
    info_list = onebyone(readme_path_list, classfication_path_list, labeling_path_list, split_pattern=" ")
    info_list = extract_data(info_list, key_value_split_pattern=":", value_split_pattern=",")
    # print(info_list)
    all_data = classification_data_info_store(info_list, "D:/数据/classification.json", "D:/数据/processinfo.txt")
    data = np.array(all_data)
    print(np.unique(data[:, 2]))
    # for info in info_list:
    #     print(info)
    #     df = pd.read_json(info["classification_path"],orient="records",encoding="utf-8",lines=True)
    #     print(df.columns)
    #     print(df["annotations"])
    #     df['label'] = df[df['annotations'].map(lambda x: len(x) > 0)]['annotations'].map(lambda x: x[0]['label'])
    #     df = df.fillna(0)
    #     df['label'] = df['label'].astype(int)
    #     df = df[["id","text","label"]]
    #     print(df)
    #     break

if __name__ == '__main__':
    main()    