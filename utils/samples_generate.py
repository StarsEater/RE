import logging
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import time
import re
import collections
import pandas as pd
from configparser import ConfigParser
from tools import *

def transfer_from_dir(text_ann_dir,save_json_path,test=False,addFake=True):
    """
    :param text_ann_dir: 存放txt和ann标注文件的路径
    :return: 该路径下的文件名集合
    """
    assert os.path.isdir(text_ann_dir),print(text_ann_dir)
    names = []
    names_list = [x.split(".")[0] for x in os.listdir(text_ann_dir) if x.endswith("txt") or x.endswith("ann")]
    assert len(names_list)!=0,print(" dir{} has not valid file".format(text_ann_dir))
    for k,v in collections.Counter(names_list).items():
        if v==2:
            names.append(k)
        else:
            print("no match file for {}".format(k))
    res = []
    names = sorted(names)[:]
    for name in names:
        txt_from = readText(os.path.join(text_ann_dir,name+".txt"),lines=False)
        ann_from = readText(os.path.join(text_ann_dir,name+".ann"),lines=True)
        entitys = list(filter(lambda x:x[0]=="T",ann_from))

        ######## from2tos, BIO, BIOS的生成   ################
        txt_pair_type = [[x,"O"] for x in txt_from]
        from2tos = []
        T2entity = {}
        for i,entity in enumerate(entitys):
            entity = entity.split("\t")
            T_num = entity[0]
            type_pos,word = entity[1:]
            en_type,start,end = type_pos.split()
            start,end = int(start),int(end)
            assert txt_from[start:end]==word,\
                print("{0} don't match {1}".format(word,txt_from[start:end]))
            T2entity[T_num] = (en_type,start,end)
            from2tos.append([start,end-1,en_type,txt_from[start:end]])
        txt_pair_type = [(re.sub(r"\t|\n|\r"," ",x[0]),x[1]) for x in txt_pair_type]
        ins,outs = list(zip(*txt_pair_type))

        #BIO
        BIO = ['O']*len(outs)
        for start,end,label_name,_ in from2tos:
            BIO[start] = 'B-'+label_name
            for i in range(start+1,end+1):
                BIO[i] = 'I-'+label_name

        # BMES
        BMES = ['O']*len(outs)
        for start,end,label_name,_ in from2tos:
            if start==end:
                BMES[start] = "S-"+label_name
                continue
            BMES[start] = 'B-'+label_name
            for i in range(start+1,end):
                BMES[i] = "M-"+label_name
            BMES[end] = "E-"+label_name

        ####### rels的输入生成 ###################
        """
        T1      first_level 0 2 右肺
        T2      second_level 2 4        下叶
        T3      third_level 4 8 前基底段
        R1      from Arg1:T3 Arg2:T2
        R2      from Arg1:T2 Arg2:T1
        """
        rels_out = []
        if not test:
            rels = list(filter(lambda x: x[0] == "R", ann_from))
            for r in rels:
                rel,arg1,arg2 = r.split("\t")[1].split(" ")
                arg1,arg2 = arg1.split(":")[1],arg2.split(":")[1]
                assert arg1 in T2entity,print("{} not in T2entity".format(arg1))
                assert arg2 in T2entity,print("{} not in T2entity".format(arg2))
                arg1,arg2 = T2entity[arg1],T2entity[arg2]
                rels_out.append([rel,arg1,arg2])
        # 添加一些反例
        if not test and addFake:
            alls_entity = list(T2entity.items())
            first_levels = list(filter(lambda x:x[-1][0]=="first_level",alls_entity))
            second_levels =list(filter(lambda x:x[-1][0]=="second_level",alls_entity))
            third_levels = list(filter(lambda x:x[-1][0]=="third_level",alls_entity))
            # if len(rels_out) > 0:
            #     print(rels_out[0],first_levels,second_levels)
            #     raise KeyboardInterrupt
            for third_l in third_levels:
                for second_l in second_levels:
                    if ['from',third_l[-1],second_l[-1]] not in rels_out:
                        rels_out.append(['O',third_l[-1],second_l[-1]])
            for third_l in third_levels:
                for first_l in first_levels:
                    if ['from',third_l[-1],first_l[-1]] not in rels_out:
                        rels_out.append(['O',third_l[-1],first_l[-1]])
            for second_l in second_levels:
                for first_l in first_levels:
                    if ['from',second_l[-1],first_l[-1]] not in rels_out:
                        rels_out.append(['O',second_l[-1],first_l[-1]])
        elif  test:
            alls_entity = list(T2entity.items())
            first_levels = list(filter(lambda x: x[-1][0] == "first_level", alls_entity))
            second_levels = list(filter(lambda x: x[-1][0] == "second_level", alls_entity))
            third_levels = list(filter(lambda x: x[-1][0] == "third_level", alls_entity))
            for third_l in third_levels:
                for second_l in second_levels:
                    rels_out.append(['O',third_l[-1],second_l[-1]])
            for third_l in third_levels:
                for first_l in first_levels:
                    rels_out.append(['O',third_l[-1],first_l[-1]])
            for second_l in second_levels:
                for first_l in first_levels:
                    rels_out.append(['O',second_l[-1],first_l[-1]])
        res.append({
            "tokens":txt_pair_type,
            "ins":ins,
            "rels":rels_out,
            "outs":outs,
            'from2tos':from2tos,
            'BIO':BIO,
            'BMES':BMES,
            "T2entity":T2entity
        })
    print("total data num is {}".format(len(res)))
    logging.info("total data num is {}".format(len(res)))
    if not test:
        checkFileOMake(save_json_path)
        save_json_path = os.path.join(save_json_path,"text_ann.json")
    saveJson(res, save_json_path,lines=True)

    print("save path is {}".format(save_json_path))
    return names,res



if __name__ == '__main__':
    config = ConfigParser()
    config_path = "../data/dev/marker_re.conf"
    # config.read(config_path,encoding='utf-8')
    config.read(config_path,encoding='utf-8')
    # print(config.sections())
    ann_dir = config['samples_generate']["ann_dir"]
    sample_save_path = config["samples_generate"]["sample_save_path"]
    log_path = config["samples_generate"]["log"]

    checkFileOMake(log_path)

    logging.basicConfig(level=logging.DEBUG,
                        filemode='a')
    logging.info("RE samples generating !")
    localtime = time.asctime(time.localtime(time.time()))
    logging.info("### start time : %s"%(localtime))
    time_stamp = int(time.time())
    logging.info("time stamp: %d"%(time_stamp))



    # print("原始数据目录: %s" % (ann_dir))
    # print("保存数据路径: %s"%(sample_save_path))
    logging.info("原始数据路径: %s"%(sample_save_path))
    logging.info("保存数据路径: %s"%(sample_save_path))
    transfer_from_dir(ann_dir, sample_save_path)





