'''
Created on 2019年7月19日

@author: gaojiexcq
'''
import os
import os.path as ospath
import pandas as pd
import numpy as np
import json

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
                    raise RuntimeError("get_all_file function Error  :"+ospath.join(base_dir, filename))
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
    for classfication_path in classfication_path_list:
        info_dict = {"classification_path":classfication_path}
        classfication_name = ospath.basename(classfication_path).rsplit(".",1)[0]
        classfication_dir = ospath.dirname(classfication_path)
        for readme_path in readme_path_list:
            readme_name = ospath.basename(readme_path).rsplit(".",1)[0]
            readme_dir = ospath.dirname(readme_path)
            if readme_name.split(split_pattern)[-1]==classfication_name.split(split_pattern)[-1] and readme_dir==classfication_dir:
                info_dict["readme_path"] = readme_path
                break
        for labeling_path in labeling_path_list:
            labeling_name = ospath.basename(labeling_path).rsplit(".",1)[0]
            labeling_dir = ospath.dirname(labeling_path)
            if classfication_name.split(split_pattern)[-1]==labeling_name.split(split_pattern)[-1] and classfication_dir==labeling_dir:
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
                raise RuntimeError("extract_data function:file {} is not utf-8".format(info_dict["readme_path"]))
                # raise(RuntimeError("file encode not is utf-8"))
        info_list[i] = info_dict
    return info_list


def classification_data_info_store(info_list, jsonpath, processjson_info_path, depreated_text="DirtyDeedsDoneDirtCheap"):
    # data = pd.DataFrame(columns=['annotation_approver','annotations','id','meta','text'])
    all_data = pd.DataFrame(columns=['id', 'text', 'label'])
    for info in info_list:
        try:
            df = pd.read_json(info["classification_path"], orient='records', encoding="utf-8", lines=True)
        except KeyError as ex:
            raise RuntimeError("classification_data_info_store function Error:{}     ".format(ex) + info["readme_path"])
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
                raise RuntimeError(" classification_data_info_store function Error:id={},info=".format(dfdata[i, 0]) + str(info))
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
            raise RuntimeError("classification_data_info_store function Error:{},info=".format(e) + str(info))
#         print(info["not negative"],info["negative but not material"],info["negetive and material"],label_to)
#         print(df.columns)
        df['label'] = df['label'].astype(np.int32)  # 恢复label为int型
        df['label'].replace(label_to, inplace=True)
        all_data = pd.concat([all_data, df], axis=0)
    all_data.to_json(jsonpath, orient="records", lines=True, force_ascii=False)
    if depreated_text is not None and depreated_text != "":
        all_data["text"].replace(depreated_text + "(:|：)*([0-9]*)", "", regex=True, inplace=True)
    data = np.array(all_data)
    data = np.delete(data, np.where(data[:, 1] == "")[0].reshape(-1), axis=0)
    delete_index = []
    for i in range(data.shape[0]):
        if data[i, 2] not in [0, 1, 2]:
            delete_index.append(i)
    data = np.delete(data, delete_index, axis=0)
    labels = np.unique(data[:, 2])
    processjson_info_str = "labels:" + str(labels) + "\n" + "classification num:" + str(data.shape[0]) + "\n"
    for lb in labels:
        percent = np.sum(data[:, 2] == lb)/data.shape[0]
        if lb == 0:
            processjson_info_str += "not negative:{:f} \n".format(percent)
        elif lb == 1:
            processjson_info_str += "negative but not material:{:f} \n".format(percent)
        elif lb == 2:
            processjson_info_str += "negative and material:{:f} \n".format(percent)
        else:
            processjson_info_str += "error:{:f} \n".format(percent)
    with open(processjson_info_path,mode="w", encoding="utf-8") as f:
        f.write(processjson_info_str)
    return all_data


def ner_data_info_restore(info_list, jsonPath, infoPath, isCharSplit=False):
    all_ner_data = None
    nerinfo = {"name": 0, "organization": 0, "when": 0, "where": 0}
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
            df = pd.read_json(info["labeling_path"],orient="records", encoding="utf-8", lines=True)
        except:
            raise RuntimeError("ner_data_info_store function json read Error: read info['labeling_path'] failed,info=" +
                               str(info))
            # raise RuntimeError("json read Error")
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
                    raise RuntimeError("ner_data_info_store function labels len get Error: id={},labels={},info="
                                       .format(data[i, 0], data[i, 2]) + str(info))
                    # raise RuntimeError("labels len get Error")
            data = np.delete(data, delete_index, axis=0)
            for i in range(data.shape[0]):
                id, text, entities = data[i, 0], data[i,1], data[i, 2]
                sorted(entities, key=(lambda x: x[0]))
                if isCharSplit:
                    '''字符分割开始'''
                    try:
                        nertext, nerlabel, nerinfo = nerTextAndLabelForCharSplit(entities, text=text,
                                                                                 need_get_entityType=False,
                                                                                 nerinfo=nerinfo)
                    except:
                        raise RuntimeError("ner_data_info_store function nertext,label get Error: id={},info="
                                           .format(id) + str(info))
                        # raise RuntimeError("nertext,label get Error")
                    '''字符分割结束'''
                else:
                    '''空格分割开始'''
                    try:
                        nertext, nerlabel, nerinfo = nerTextAndLabelForSpaceSplit(entities, text=text,
                                                                                  need_get_entityType=False,
                                                                                  nerinfo=nerinfo)
                    except:
                        raise RuntimeError("ner_data_info_store function nertext,label get Error: id={},info="
                                           .format(id) + str(info))
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
                    raise RuntimeError("ner_data_info_store function annotations len get Error: id={}," +
                                       "annotations={},info={}".format(data[i, 0], data[i, 2], info))
                    # raise RuntimeError("annotations len get Error")
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
                    raise RuntimeError("ner_data_info_stroe function annotations entity get Error:id={},info={}"
                                       .format(id, info))
                    # raise RuntimeError("annotations entity get Error")
                if isCharSplit:
                    '''字符分割做法开始'''
                    try:
                        nertext, nerlabel, nerinfo = nerTextAndLabelForCharSplit(entities, namelabels, organizationlabels,
                                                                                 whenlabels, wherelabels, text,
                                                                                 nerinfo=nerinfo)
                    except:
                        raise RuntimeError("ner_data_info_store function nertext,label get Error: id={},info="
                                           .format(id) + str(info))
                        # print("nertext,label get Error: id=", id, "info=", info)
                        # raise RuntimeError("nertext,label get Error")
                    '''字符分割做法结束'''
                else:
                    '''空格分割做法开始'''
                    try:
                        nertext, nerlabel, nerinfo = nerTextAndLabelForSpaceSplit(entities, namelabels,
                                                                                  organizationlabels, whenlabels,
                                                                                  wherelabels, text,
                                                                                  nerinfo=nerinfo)
                    except:
                        raise RuntimeError("ner_data_info_store function nertext,label get Error: id={},info="
                                           .format(id) + str(info))
                        # print("nertext,label get Error: id=", id, "info=", info)
                        # raise RuntimeError("nertext,label get Error")
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
    all_data_df.to_json(jsonPath, orient="records", force_ascii=False, lines=True)
    count_max_512 = 0
    for d in all_ner_data:
        if len(d[1]) > 512:
            count_max_512 += 1
    with open(infoPath, mode="w+", encoding="utf-8") as f:
        f.write(json.dumps(nerinfo)+"\ncount_max_512={}".format(count_max_512))
    # print("count_max_512=", count_max_512)
    return all_ner_data, count_max_512, nerinfo

def nerTextAndLabelForCharSplit(entities, namelabels=None, organizationlabels=None, whenlabels=None, wherelabels=None,
                                           text=None, need_get_entityType=True, nerinfo=None):
    textsplit = list(text)
    label = []
    start_offset = 0
    for entity in entities:
        if need_get_entityType:
            labelname = entityType(namelabels, organizationlabels, whenlabels, wherelabels, entity[2])
        else:
            labelname = entity[2].lower()
            if labelname == "person name":
                labelname = "name"
            elif labelname == "organization name":
                labelname = "organization"
            elif labelname == "where:":
                labelname = "where"
            elif labelname == "when:":
                labelname = "when"
            else:
                pass
            if labelname not in ["name", "organization", "where", "when"]:
                raise RuntimeError("labels provided but not in name organization where when"+
                                   " your labelname is" + labelname)
        if labelname != "none":
            for i in range(start_offset,int(entity[0]), 1):
                label.append("O")
            for i in range(int(entity[0]),int(entity[1]),1):
                if i==int(entity[0]):
                    label.append("B-"+labelname)
                elif i<int(entity[1])-1:
                    label.append("I-"+labelname)
                else:
                    label.append("E-"+labelname)
            if nerinfo is not None:
                nerinfo[labelname] += 1
        else:
            for i in range(start_offset,int(entity[1]),1):
                label.append("O")
        start_offset = int(entity[1])
    while start_offset<len(text):
        label.append("O")
        start_offset += 1
    return textsplit, label, nerinfo

def nerTextAndLabelForSpaceSplit(entities, namelabels=None, organizationlabels=None, whenlabels=None, wherelabels=None,
                                           text=None, need_get_entityType=True, nerinfo=None):
    textsplit = []
    label = []
    start_offset = 0
    for entity in entities:
        if need_get_entityType:
            labelname = entityType(namelabels, organizationlabels, whenlabels, wherelabels, entity[2])
        else:
            labelname = entity[2].lower()
            if labelname == "person name":
                labelname = "name"
            elif labelname == "organization name":
                labelname = "organization"
            elif labelname == "where:":
                labelname = "where"
            elif labelname == "when:":
                labelname = "when"
            else:
                pass
            if labelname not in ["name", "organization", "where", "when"]:
                raise RuntimeError("labels provided but not in name organization where when"+
                                   " your labelname is " + labelname)
        if labelname != "none":
            try:
                temp = text[start_offset:int(entity[0])]
            except:
                print("entity start offset error:entity=", entity)
                raise RuntimeError("entity start offset error")
            if len(temp.strip()) > 0:
                tempsplit = temp.strip().split(" ")
                for tmp in tempsplit:
                    textsplit.append(tmp)
                    label.append("O")
            temp = text[int(entity[0]):int(entity[1])]
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
            if nerinfo is not None:
                nerinfo[labelname] += 1
        else:
            tempsplit = text[start_offset:int(entity[1])].strip().split(" ")
            for tmp in tempsplit:
                textsplit.append(tmp)
                label.append("O")
        start_offset = int(entity[1])
    if start_offset < len(text):
        tempsplit = text[start_offset:].strip().split(" ")
        for tmp in tempsplit:
            textsplit.append(tmp)
            label.append("O")
    return textsplit, label, nerinfo


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


def count_len_with_word(classification_path,ner_path):
    df = pd.read_json(classification_path,orient="records", lines=True)
    max_len = None
    min_len = None
    all_len = 0
    data = np.array(df["text"])
    count = data.shape[0]
    count_max_200 = 0
    for i in range(data.shape[0]):
        length = len(data[i].split(" "))
        if max_len is None:
            max_len = length
        else:
            if length > max_len:
                max_len = length
        if min_len is None:
            min_len = length
        else:
            if length < min_len:
                min_len = length
        all_len += length
        if length > 200:
            count_max_200 += 1

    df = pd.read_json(ner_path, orient="records", lines=True)
    data = np.array(df["text"])
    count += data.shape[0]
    for i in range(data.shape[0]):
        if isinstance(data[i], list):
            length = len(data[i])
        elif isinstance(data[i],np.ndarray):
            length = data[i].shape[0]
        else:
            length = len(data[i].split(" "))
        if max_len is None:
            max_len = length
        else:
            if length > max_len:
                max_len = length
        if min_len is None:
            min_len = length
        else:
            if length < min_len:
                min_len = length
        all_len += length
        if length > 200:
            count_max_200 += 1
    avg_len = all_len // count
    print("min_len=",min_len,",max_len=",max_len)
    print("avg_len=", avg_len, ",count_max_200=",count_max_200, ",count_min_200=",count-count_max_200)



def JD_process(tag_file_list, classification_target_file, labeling_target_file, nerinfo_file, classificationinfo_file,
               isCharSplit=False):
    jd_classification_data = []
    jd_ner_data = []
    classification_label = ["not negative", "negative but not material", "negative and material"]
    nerinfo = {"name": 0, "organization": 0, "when": 0, "where": 0}
    for file in tag_file_list:
        df = pd.read_json(file, "records", encoding="utf-8", lines=True)
        # print(df)
        data = np.array(df[["id","text","annotations"]])
        for i in range(data.shape[0]):
            entities = []
            for annotation in data[i,2]:
                # print(annotation["label"], annotation["label"].strip() in classification_label)
                if annotation["label"].strip() in classification_label:
                    jd_classification_data.append(np.array([data[i,0],data[i,1],
                                                            classification_label.index(annotation["label"])]))
                else:
                    entities.append([annotation["start_offset"], annotation["end_offset"], annotation["label"]])
            id, text = data[i,0], data[i,1]
            if len(entities)>0:
                sorted(entities, key=(lambda x: x[0]))
                if isCharSplit:
                    '''字符分割开始'''
                    nertext, nerlabel, nerinfo = nerTextAndLabelForCharSplit(entities, text=text,
                                                                             need_get_entityType=False,
                                                                             nerinfo=nerinfo)
                    '''字符分割结束'''
                else:
                    '''空格分割开始'''
                    nertext, nerlabel, nerinfo = nerTextAndLabelForSpaceSplit(entities, text=text,
                                                                              need_get_entityType=False,
                                                                              nerinfo=nerinfo)
                    '''空格分割结束'''
                jd_ner_data.append(np.array([id, nertext, nerlabel]))
    jd_classification_data = np.array(jd_classification_data)
    jd_ner_data = np.array(jd_ner_data)
    jd_classification_data = np.delete(jd_classification_data,
                                       np.where(jd_classification_data[:, 1] == "")[0].reshape(-1), axis=0)
    delete_index = []
    # print(jd_classification_data)
    for i in range(jd_classification_data.shape[0]):
        if int(jd_classification_data[i, 2]) not in [0, 1, 2]:
            delete_index.append(i)
    jd_classification_data = np.delete(jd_classification_data, delete_index, axis=0)
    jd_classification = pd.DataFrame(jd_classification_data,columns=["id","text","label"])
    jd_ner = pd.DataFrame(jd_ner_data, columns=["id","text","label"])
    jd_classification.to_json(classification_target_file,"records",lines=True,force_ascii=False)
    jd_ner.to_json(labeling_target_file, "records", lines=True, force_ascii=False)

    labels = np.unique(jd_classification_data[:, 2])
    classification_info_str = "labels:" + str(labels) + "\n" + "classification num:" + str(data.shape[0]) + "\n"
    for lb in labels:
        percent = np.sum(jd_classification_data[:, 2] == lb) / jd_classification_data.shape[0]
        lb = int(lb)
        if lb == 0:
            classification_info_str += "not negative:{:f} \n".format(percent)
        elif lb == 1:
            classification_info_str += "negative but not material:{:f} \n".format(percent)
        elif lb == 2:
            classification_info_str += "negative and material:{:f} \n".format(percent)
        else:
            classification_info_str += "error:{:f} \n".format(percent)
    with open(classificationinfo_file, mode="w", encoding="utf-8") as f:
        f.write(classification_info_str)

    count_max_512 = 0
    for d in jd_ner_data:
        if len(d[1]) > 512:
            count_max_512 += 1
    with open(nerinfo_file, mode="w+", encoding="utf-8") as f:
        f.write(json.dumps(nerinfo) + "\ncount_max_512={}".format(count_max_512))
    return jd_classification, jd_ner, nerinfo


def main():
    jd_base_dir = "D:/数据/jd_dataset/ch"
    tag_file_list = get_all_file(jd_base_dir, "labeling", "json", True)
    JD_process(tag_file_list, jd_base_dir+"/classification.json", jd_base_dir+"/ner.json",jd_base_dir+"/nerinfo.txt",
               jd_base_dir+"/classificationinfo.txt",isCharSplit=True)


    base_dir = "D:/数据/dataset"
    readme_path_list = get_all_file(base_dir, "readme", "txt", True)
    classfication_path_list = get_all_file(base_dir, "classification", "json", True)
    labeling_path_list = get_all_file(base_dir, "labeling", "json", True)
    info_list = onebyone(readme_path_list, classfication_path_list, labeling_path_list, split_pattern=" ")
    info_list = extract_data(info_list, key_value_split_pattern=":", value_split_pattern=",")
    ## for  classification
    all_data = classification_data_info_store(info_list, "D:/数据/数据处理结果/data/classification.json",
                                              "D:/数据/数据处理结果/info/classificationinfo/classificationinfo0731.txt")
    data = np.array(all_data)



    ##for ner
    all_ner_data, count_max_512 = ner_data_info_restore(info_list, "D:/数据/数据处理结果/data/ner.json",
                                                        "D:/数据/数据处理结果/info/nerinfo/nerinfo0731.txt",
                                                        isCharSplit=False)
    print("ner text split with space or char and count_max_512=", count_max_512)
    count_len_with_word("D:/数据/数据处理结果/data/classification.json", "D:/数据/数据处理结果/data/ner.json")


if __name__ == '__main__':
    main()
