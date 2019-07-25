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
    all_data.to_json(jsonpath, orient="records", lines=True,force_ascii=False)
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


def ner_data_info_restore(info_list, jsonpath, isCharSplit=False):
    all_ner_data = None
    for info in info_list:
        '''获取readme文件中的ner标签对应数组 start'''
        if "name" in info:
            namelabels = info["name"]
        elif "person name" in info:
            namelabels = info["person name"]
        else:
            namelabels = []
        if "organization" in info:
            organizationlabels = info["organization"]
        elif "organization name" in info:
            organizationlabels = info["organization name"]
        else:
            organizationlabels = []
        if "when" in info:
            whenlabels = info["when"]
        else:
            whenlabels = []
        if "where" in info:
            wherelabels = info["where"]
        else:
            wherelabels = []
        '''获取readme文件中的ner标签对应数组 end'''

        try:
            df = pd.read_json(info["labeling_path"],orient="records",encoding="utf-8",lines=True)
        except:
            print("json read Error: read info['labeling_path'] failed", info)
            raise RuntimeError("json read Error")
        ner_data = None
        if "labels" in df.columns:
            '''新版本做法'''
            columns = ["id", "text", "labels"]
            data = np.array(df[columns])
            delete_index = []
            for i in range(data.shape[0]):
                try:
                    if len(data[i, 2]) <= 0:
                        delete_index.append(i)
                except:
                    print("labels len get Error: id=", data[i, 0], "labels=", data[i, 2], "info=", info)
                    raise RuntimeError("labels len get Error")
            data = np.delete(data, delete_index, axis=0)
            for i in range(data.shape[0]):
                id, text, entities = data[i, 0], data[i,1], data[i, 2]
                sorted(entities, key=(lambda x: x[0]))
                if isCharSplit:
                    '''字符分割开始'''
                    try:
                        nertext,nerlabel = nerTextAndLabelForCharSplit(entities,text=text,need_get_entityType=False)
                    except:
                        print("nertext,label get Error: id=", id, "info=", info)
                        raise RuntimeError("nertext,label get Error")
                    '''字符分割结束'''
                else:
                    '''空格分割开始'''
                    try:
                        nertext,nerlabel = nerTextAndLabelForSpaceSplit(entities, text=text, need_get_entityType=False)
                    except:
                        print("nertext,label get Error: id=", id, "info=", info)
                        raise RuntimeError("nertext,label get Error")
                    '''空格分割结束'''
                if ner_data is None:
                    ner_data = np.array([[id, nertext, nerlabel]])
                else:
                    ner_data = np.r_[ner_data, np.array([[id, nertext, nerlabel]])]
        else:
            '''在readme里面有标签提示的做法'''
            columns = ["id", "text", "annotations"]
            data = np.array(df[columns])
            delete_index = []
            for i in range(data.shape[0]):
                try:
                    if len(data[i, 2]) <= 0:
                        delete_index.append(i)
                except:
                    print("annotations len get Error: id=", data[i, 0], "annotations=", data[i,2], "info=", info)
                    raise RuntimeError("annotations len get Error")
            data = np.delete(data, delete_index, axis=0)
            for i in range(data.shape[0]):
                id = data[i, 0]
                text = data[i, 1]
                annotations = data[i, 2]
                # namelabels, organizationlabels, whenlabels, wherelabels = [], [], [], []
                try:
                    entities = []
                    '''获取标签数值及该标签在文本中的位置偏移开始与结束'''
                    for annotation in annotations:
                        entity = []
                        entity.append(annotation["start_offset"])
                        entity.append(annotation["end_offset"])
                        entity.append(annotation["label"])
                        entities.append(np.array(entity).tolist())
                    sorted(entities, key=(lambda x: x[0]))
                except:
                    print("annotations entity get Error:id=", id, "info=", info)
                    raise RuntimeError("annotations entity get Error")
                if isCharSplit:
                    '''字符分割做法开始'''
                    try:
                        nertext,nerlabel = nerTextAndLabelForCharSplit(entities, namelabels, organizationlabels,
                                                                                  whenlabels,wherelabels, text)
                    except:
                        print("nertext,label get Error: id=", id, "info=", info)
                        raise RuntimeError("nertext,label get Error")
                    '''字符分割做法结束'''
                else:
                    '''空格分割做法开始'''
                    try:
                        nertext, nerlabel = nerTextAndLabelForSpaceSplit(entities, namelabels, organizationlabels, whenlabels,
                                                                         wherelabels, text)
                    except:
                        print("nertext,label get Error: id=", id, "info=", info)
                        raise RuntimeError("nertext,label get Error")
                    '''空格分割做法结束'''
                if ner_data is None:
                    ner_data = np.array([[id, nertext, nerlabel]])
                else:
                    ner_data = np.r_[ner_data, np.array([[id, nertext, nerlabel]])]
            # print(ner_data)
                # break
        if all_ner_data is None:
            all_ner_data = ner_data
        else:
            all_ner_data = np.r_[all_ner_data, ner_data]
        # break
    all_data_df = pd.DataFrame(all_ner_data,columns=["id", "text", "label"])
    all_data_df.to_json(jsonpath, orient="records", force_ascii=False, lines=True)
    count_max_512 = 0
    for d in all_ner_data:
        if len(d[1]) > 512:
            count_max_512 += 1
    # print("count_max_512=", count_max_512)
    return all_ner_data, count_max_512

def nerTextAndLabelForCharSplit(entities, namelabels=None, organizationlabels=None, whenlabels=None, wherelabels=None,
                                           text=None, need_get_entityType=True):
    textsplit = list(text)
    label = []
    start_offset = 0
    for entity in entities:
        if need_get_entityType:
            labelname = entityType(namelabels, organizationlabels, whenlabels, wherelabels, entity[2])
        else:
            labelname = entity[2].lower()
        if labelname != "none":
            for i in range(start_offset,entity[0],1):
                label.append("O")
            for i in range(entity[0],entity[1],1):
                if i==entity[0]:
                    label.append("B-"+labelname)
                elif i<entity[1]-1:
                    label.append("I-"+labelname)
                else:
                    label.append("E-"+labelname)
        else:
            for i in range(start_offset,entity[1],1):
                label.append("O")
        start_offset = entity[1]
    while start_offset<len(text):
        label.append("O")
        start_offset += 1
    return textsplit, label

def nerTextAndLabelForSpaceSplit(entities, namelabels=None, organizationlabels=None, whenlabels=None, wherelabels=None,
                                           text=None, need_get_entityType=True):
    textsplit = []
    label = []
    start_offset = 0
    for entity in entities:
        if need_get_entityType:
            labelname = entityType(namelabels, organizationlabels, whenlabels, wherelabels, entity[2])
        else:
            labelname = entity[2].lower()
        if labelname != "none":
            try:
                temp = text[start_offset:entity[0]]
            except:
                print("entity start offset error:entity=", entity)
                raise RuntimeError("entity start offset error")
            if len(temp.strip()) > 0:
                tempsplit = temp.strip().split(" ")
                for tmp in tempsplit:
                    textsplit.append(tmp)
                    label.append("O")
            temp = text[entity[0]:entity[1]]
            if len(temp.strip()) > 0:
                tempsplit = temp.strip().split(" ")
                if len(tempsplit) > 1:
                    for i in range(len(tempsplit)):
                        textsplit.append(tempsplit[i])
                        if i == 0:
                            label.append("B-" + labelname)
                        elif i < len(tempsplit) - 1:
                            label.append("I-" + labelname)
                        else:
                            label.append("E-" + labelname)
                else:
                    textsplit.append(tempsplit[0])
                    label.append("S-" + labelname)
        else:
            tempsplit = text[start_offset:entity[1]].strip().split(" ")
            for tmp in tempsplit:
                textsplit.append(tmp)
                label.append("O")
        start_offset = entity[1]
    if start_offset < len(text):
        tempsplit = text[start_offset:].strip().split(" ")
        for tmp in tempsplit:
            textsplit.append(tmp)
            label.append("O")
    return textsplit, label


def entityType(namelabels, organizationlabels, whenlabels, wherelabels, target):
    if target in namelabels:
        return "name"
    elif target in organizationlabels:
        return "organization"
    elif target in whenlabels:
        return "when"
    elif target in wherelabels:
        return "where"
    else:
        return "none"


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
    ## for  classification
    all_data = classification_data_info_store(info_list, "D:/数据/classification.json", "D:/数据/classificationinfo.txt")
    data = np.array(all_data)
    print("classification label=", np.unique(data[:, 2]))

    ##for ner
    all_ner_data,count_max_512 = ner_data_info_restore(info_list, "D:/数据/ner.json", isCharSplit=False)
    print("ner text split with space or char and count_max_512=", count_max_512)


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