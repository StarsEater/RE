"""label mask must make sure other - > 0"""
import collections
import logging
import os
import random
import sys
import time
import torch.nn as nn
import torchnet.meter as meter
from configparser import ConfigParser

import torch
import numpy as np
import tqdm
from prettyprinter import cpprint
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.marker_re.dataset import marker_re_dataset
from model.marker_re_model import marker_re_model
from tools import *
from utils._utils import plot_loss, plot_f1_score, modify_bert_tokenizer
from transformers import BertTokenizer,AdamW, get_linear_schedule_with_warmup

import warnings

from utils.dataset_merge_and_shuffle import dataset_merge_shuffle
from utils.metrics import one_label_f1_score
from utils.samples_generate import transfer_from_dir

warnings.filterwarnings("ignore")
task_name = os.path.abspath(os.path.dirname(os.path.dirname(__file__))).split("/")[-1]
def seed_torch(seed=2200):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()

def PreProcess(train_raw_path,labels_map,config_path = "./data/dev/train_pipeline.conf",project_name=None):
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')
    dataset_save_path = config['dataset_split']['dataset_save_path'].replace("Pathology",project_name)
    pass


def data_initialization(data, train_file, dev_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.fix_alphabet()
    print(data.word_alphabet_size)
    print(data.biword_alphabet_size)
    return data

def Train(config_path = "./data/dev/marker_re.conf"):
    """
    :param labels:  标签列表
    :param clss:  类别
    :param config_path: 配置文件名
    :return:
    """
    config = ConfigParser()
    config.read(config_path,encoding='utf-8')
    dataset_save_path = config['dataset_split']['dataset_save_path']

    device_id = config['model_train']['train_device_id']
    device_id = [int(item) for item in device_id.split(",")]
    trainset_path    = os.path.join(dataset_save_path,config['model_train']['train_path'])
    valset_path      = os.path.join(dataset_save_path,config['model_train']['dev_path'])
    resume           = int(config['model_train']['resume'])
    checkpoint_path  = config['model_train']['checkpoint']
    history_path     = config['model_train']['history']
    log_path         = config['model_train']['log']

    model_choice     = config['model_train']['model_choice']
    model_name       = config['model_train']['model_save_name']
    model_resume_name= config['model_train']['model_resume_name']

    bert_path        = config['model_train']['bert_path']
    use_gpu          = eval(config['model_train']['use_gpu'])
    end_epoch        = eval(config['model_train']['end_epoch'])
    batch_size       = eval(config['model_train']['batch_size'])
    lr               = eval(config['model_train']['lr'])
    warmup           = eval(config['model_train']['warmup'])
    num_warmup_steps = eval(config['model_train']['num_warmup_steps'])
    num_total_steps  = eval(config['model_train']['num_total_steps'])



    # model_setting
    # labels
    rel_labels        = config['samples_generate']['rels_label'].split(",")
    entity_label_lst  = config['samples_generate']['entity_label'].split(",")

    checkFileOMake(log_path)
    checkFileOMake(checkpoint_path)
    checkFileOMake(history_path)

    log_save_name = 'log_' + model_name + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(log_path,log_save_name),
                        filemode='a')
    checkpoint_name = os.path.join(checkpoint_path,model_name+'_best_ckpt.pth')
    model_ckpt_name = os.path.join(checkpoint_path,model_name+'_best.pkl')
    print("模型保存路径为%s"%(model_ckpt_name))
    print("模型训练评价指标保存路径为%s"%(checkpoint_name))
    checkFileOMake(model_resume_name)

    localtime = time.asctime(time.localtime(time.time()))
    logging.info('#### start time : %s'%(localtime))
    time_stamp = int(time.time())
    logging.info('time stamp: %d'%(time_stamp))
    logging.info('###### Model: %s'%(model_name))
    logging.info('trainset path: %s'%(trainset_path))
    logging.info('valset path: %s'%(valset_path))

    ###############  rel2num 和  tokenizer 的初始化 ########################
    rel2num = {v:i for i,v in enumerate(rel_labels)}
    tokenizer = modify_bert_tokenizer(path=bert_path,entity_label_lst=entity_label_lst)


    #######################################
    if model_choice=='pretrained':
        pass
    elif model_choice=='bert':
        print("loading <bert> dataset    ... ...")
        train_dataloader = marker_re_dataset(path=trainset_path,tokenizer=tokenizer,rel2num=rel2num,normalize_num=True).\
            get_dataloader(batch_size,num_workers=0,shuffle=False,pin_memory=False)
        val_dataloader = marker_re_dataset(path=trainset_path, tokenizer=tokenizer, rel2num=rel2num, normalize_num=True). \
            get_dataloader(batch_size, num_workers=0, shuffle=True, pin_memory=False)

    logging.info('batch_size:%d'%(batch_size))
    logging.info('learning rate:%f'%(lr))
    logging.info('iteration:%d'%(end_epoch))
    device= torch.device('cuda:'+str(device_id[0]) if torch.cuda.is_available() else 'cpu')

    print("use",device)

    #####################################
    if model_choice=='bert':
        model = marker_re_model(bert_path,label_num=len(rel_labels))
    elif model_choice=='':
        pass

    if len(device_id) > 1:
        print("let's use ",len(device_id),'GPU!')
        model = nn.DataParallel(model,device_id=device_id)

    model.to(device)

   ######################################
    if resume !=0:
        logging.info("Resuming from checkpoint ...")
        model.load_state_dict(torch.load(model_resume_name))
        checkpoint = torch.load(checkpoint_name)
        best_f1 = checkpoint['f1']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
    else:
        best_f1 = 0.0
        start_epoch = -1
        history = {
            'train_loss':[],'val_loss':[],
            'train_f1':[],'val_f1':[],
            'train_pre':[],'val_pre':[],
            'train_rec':[],'val_rec':[]
        }

    ###################################
    bert_parameters = list(filter(lambda p:p in model.bert.named_parameters(),model.named_parameters()))
    others = list(filter(lambda p:p not in model.bert.parameters(),model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':bert_parameters,'lr':lr},
        {'params': [p for n, p in others if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in others if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if not warmup:
        optim = torch.optim.Adam(optimizer_grouped_parameters, lr=lr*5)
        scheduler = StepLR(optim, step_size=5, gamma=0.8)
    else:
        optim = AdamW(optimizer_grouped_parameters, lr=lr*5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_total_steps)  # PyTorch scheduler

        logging.info("warm up steps %d " % (num_warmup_steps))
        logging.info("total steps %d " % (num_total_steps))

    print("Start training")
    writer = SummaryWriter(os.path.join('./summary_runs/',model_name))
    global_step = 0
    loss_tr = meter.AverageValueMeter()
    f1_tr   = meter.AverageValueMeter()
    pre_tr  = meter.AverageValueMeter()
    rec_tr  = meter.AverageValueMeter()

    loss_va = meter.AverageValueMeter()
    f1_va   = meter.AverageValueMeter()
    pre_va  = meter.AverageValueMeter()
    rec_va  = meter.AverageValueMeter()

    for epoch in range(start_epoch+1,end_epoch):
        print("--------------epoch:%d------------"%(epoch))
        logging.info("-----------epoch:%d-----------"%(epoch))
        model.train()
        loss_tr.reset()
        f1_tr.reset()
        pre_tr.reset()
        rec_tr.reset()

        loss_va.reset()
        f1_va.reset()
        pre_va.reset()
        rec_va.reset()

        print("start training !!! !!!")
        for batch_id,batch in enumerate(train_dataloader):
            ins_tensor,rels,rels_pos,pos_en,padding_mask,token_type_ids,attention_mask = map(lambda x:x.to(device),batch)
            model.zero_grad()
            loss,*_ = model(ins_tensor,rels,rels_pos,pos_en,padding_mask,token_type_ids,attention_mask)
            loss.backward()

            loss_item = loss.item()
            writer.add_scalar('train_loss',loss_item,global_step)
            global_step += 1

            optim.step()

            if warmup:
                scheduler.step()
            loss_tr.add(loss_item)
            if batch_id % 20==0:
                print('-----------------batch:%d------------'%(batch_id))
                print('-----------------loss:%f--------------'%(loss_item))
        mean_loss = loss_tr.value()[0]
        print('trainset loss:%f' % (mean_loss))
        logging.info('trainset loss:%f' % (mean_loss))
        history['train_loss'].append(mean_loss)

        ###################  val ########################
        model.eval()
        with torch.no_grad():
            print("start validating !!! !!!")
            true_label_lst, pre_label_lst = [],[]
            for batch_id, batch in enumerate(val_dataloader):
                ins_tensor, rels, rels_pos, pos_en, padding_mask, token_type_ids, attention_mask = map(
                    lambda x: x.to(device), batch)
                loss, preds,rels = model(ins_tensor, rels, rels_pos, pos_en, padding_mask, token_type_ids, attention_mask)

                true_label_lst += rels.detach().tolist()
                pre_label_lst  += preds.detach().tolist()

            _,pre,rec,f1 = one_label_f1_score(true_label_lst,pre_label_lst,target_name=rel_labels)

            pre_va.add(pre)
            rec_va.add(rec)
            f1_va.add(f1)

            mean_loss = loss_va.value()[0]
            mean_pre = pre_va.value()[0]
            mean_rec = rec_va.value()[0]
            mean_f1 = f1_va.value()[0]

            logging.info('valset loss:%f' % (mean_loss))
            logging.info('valset f1:%f' % (mean_f1))
            logging.info('valset pre:%f' % (mean_pre))
            logging.info('valset rec:%f' % (mean_rec))

            history['val_loss'].append(mean_loss)
            history['val_f1'].append(mean_f1)
            history['val_pre'].append(mean_pre)
            history['val_rec'].append(mean_rec)

            if mean_f1 >best_f1:
                logging.info('Checkpoint Saving ...')
                print("best f1 score so far ! Checkpoint Saving ...")
                print("save state path is %s \nmodel path is %s"%(checkpoint_name,model_ckpt_name))
                state = {
                    'epoch':epoch,
                    'f1':mean_f1,
                    'history':history
                }
                torch.save(state,checkpoint_name)
                best_f1 = mean_f1
                torch.save(model.state_dict(),model_ckpt_name)
            plot_loss(history,history_path,model_name,time_stamp)
            plot_f1_score(history,history_path,model_name,time_stamp)
        if not warmup:
            scheduler.step()
            logging.info('current lr:%f'%(scheduler.get_lr()[0]))

@torch.no_grad()
def Test(config_path = "./data/dev/marker_re.conf"):
    print("start Test !!! !!! ")
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')

    dataset_save_path = config['dataset_split']['dataset_save_path']
    testset_path      = os.path.join(dataset_save_path,config['model_train']['test_path'])

    rel_labels = config['samples_generate']['rels_label'].split(",")
    entity_label_lst = config['samples_generate']['entity_label'].split(",")

    train_device_id = config['model_train']['train_device_id']
    train_device_id = [int(item) for item in train_device_id.split(",")]
    test_device_id = config['model_test']['test_device_id']
    test_device_id = [int(item) for item in test_device_id.split(",")]
    device = torch.device('cuda:' + str(test_device_id[0]) if torch.cuda.is_available() else 'cpu')
    train_cuda = 'cuda:' + str(train_device_id[0])
    test_cuda = 'cuda:' + str(test_device_id[0])

    bert_path = config['model_train']['bert_path']
    use_gpu = eval(config['model_train']['use_gpu'])
    model_name = config['model_train']['model_save_name']
    model_resume_name = config['model_train']['model_resume_name']
    model_ckpt_name = os.path.join(model_resume_name, model_name + '_best.pkl')
    model_choice = config['model_train']['model_choice']

    pre_out_dir = config['model_test']['pred_out_path']
    test_batch_size = eval(config['model_test']['test_batch_size'])
    checkFileOMake(pre_out_dir)

    test_out_names, ress = transfer_from_dir(pre_out_dir, testset_path, test=True)
    ###############  rel2num 和  tokenizer 的初始化 ########################
    rel2num = {v: i for i, v in enumerate(rel_labels)}
    tokenizer = modify_bert_tokenizer(path=bert_path, entity_label_lst=entity_label_lst)

    if model_choice=='pretrained':
        pass
    elif model_choice=='bert':
        print("loading <bert> dataset    ... ...")
        test_dataloader = marker_re_dataset(path=testset_path, tokenizer=tokenizer, rel2num=rel2num,
                                           normalize_num=True). \
            get_dataloader(test_batch_size, num_workers=0, shuffle=False, pin_memory=False)

    # 加载类型模型
    if model_choice == 'bert':
        model = marker_re_model(bert_path,label_num=len(rel_labels))
    elif model_choice == '':
        pass

    model.to(device)

    logging.info("Type Model Resuming from checkpoint ...")
    model.load_state_dict(torch.load(model_ckpt_name, map_location={train_cuda: test_cuda}))
    model.eval()

    pre_label_lst = []

    for batch_id, batch in enumerate(test_dataloader):
        ins_tensor, rels, rels_pos, pos_en, padding_mask, token_type_ids, attention_mask = map(
            lambda x: x.to(device), batch)
        in_pred = model(ins_tensor, rels, rels_pos, pos_en, padding_mask, token_type_ids, attention_mask,True)
        pre_label_lst += in_pred

    saveJson(pre_label_lst,os.path.join(pre_out_dir,"ma_re_test.json"),lines=True)
    # print(span_preds)

    # 写入text
    # cpprint(span_preds)

    # [type,e1,e2]
    for name,res,rel_res in zip(test_out_names,ress,pre_label_lst):
        t2entity,rels_out_raw = res["T2entity"],res["rels"]
        path_out = os.path.join(pre_out_dir,name+".ann")
        rel_pred = [rel_labels[x] for x in rel_res]
        entity2t = {v:i for i,v in t2entity.items()}
        # T2entity[T_num] = [x,(en_type, start, end),(en_type, start, end)]
        with open(path_out,"a",encoding='utf-8') as f :
            cur_i = 1
            # f.write("\n")
            for i,(r,s,p) in enumerate(rels_out_raw):
                r = rel_pred[i]
                if r=="O":
                    continue
                s = entity2t[s]
                p = entity2t[p]
                line = "R"+str(cur_i)+"\t"+r+" "+"Arg1:"+s + " "+"Arg2:"+p
                cur_i += 1
                f.write(line)
                f.write("\n")


def PostProcess(config_path = ""):
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')
    pass


def PipeLine(config_path = ""):
    # config = ConfigParser()
    # config.read(config_path, encoding='utf-8')

    print("******* PreProcess (including split labels and dataset) *********")
    from copy import deepcopy as cp

    print("*******************start train *****************************")
    # Train()
    Test()
    print('******** PostProcess (add some rules)  **********************')


#
if __name__ == '__main__':
    PipeLine()

# if __name__ == '__main__':
#     print(task_name)

