from transformers import BertTokenizer
import os
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
class modify_bert_tokenizer():
    def __init__(self,path,entity_label_lst):
         self.basic_tokenizer = BertTokenizer.from_pretrained(path,do_lower_case=True)
         self.vocab = self.basic_tokenizer.vocab
         never_split_lst = []
         for e in entity_label_lst:
             e =e.lower()
             never_split_lst +=["<s:"+str(e)+">","<o:"+str(e)+">","</s:"+str(e)+">","</o:"+str(e)+">"]
         print("add marker is {}".format(len(never_split_lst)))
         for i,token in enumerate(never_split_lst):
             self.vocab[token] = i+1
             assert i < 100

    def convert_tokens_to_ids(self, tokens):
        return self.basic_tokenizer.convert_tokens_to_ids(tokens)
    def convert_ids_to_tokens(self,ids):
        return self.basic_tokenizer.convert_ids_to_tokens(ids)
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word +='0'
        else:
            new_word +=char.lower()
    return new_word


def plot_loss(history, save_root, model_name,time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_loss'], 'r', history['val_loss'], 'b')
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss during training")
    if save_mode:
        plt.savefig(os.path.join(save_root,'loss_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()

def plot_acc_score(history, save_root,model_name, time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_acc'], 'r', history['val_acc'], 'b')
    plt.legend(["train_acc_score", "val_acc_score"])
    plt.xlabel("epoch")
    plt.ylabel("acc_score")
    plt.title("acc_score during training")
    if save_mode:
        plt.savefig(os.path.join(save_root , 'acc_score_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()

def plot_f1_score(history, save_root, model_name,time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_f1'], 'r', history['val_f1'], 'b')
    plt.legend(["train_f1_score", "val_f1_score"])
    plt.xlabel("epoch")
    plt.ylabel("f1_score")
    plt.title("f1_score during training")
    if save_mode:
        plt.savefig(os.path.join(save_root , 'f1_score_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()




if __name__ == '__main__':
    path = "/nlp_data/qinye/pretrains/RoBERTa_zh_L12_PyTorch"
    tokenizer = modify_bert_tokenizer(path,["<s:first_level>"])
    c = ["<s:first_level>","a","b"]
    cc = tokenizer.convert_tokens_to_ids(c)
    print(cc)