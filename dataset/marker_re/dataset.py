import os
from pprint import pprint

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader

from tools import loadPickle, readJson
from utils._utils import normalize_word

""""
!!!!!!!!! 确保0对应到others

"""

class marker_re_dataset(Dataset):
    def __init__(self,path,tokenizer,rel2num=None,normalize_num=True):
        super(marker_re_dataset, self).__init__()
        assert 'O' in rel2num and rel2num['O']==0,print("relnum {} is not equal 0".format('O'))
        self.process_data = []
        self.rel2num = rel2num
        self.tokenizer = tokenizer
        for d in readJson(path,lines=True):
            # d["rels"] [rel,(ent,start,end),()]
            ins,rels_out = d["ins"],d["rels"]
            if normalize_num:
                ins = list(normalize_word(ins))
            pdata = self.convert_to_marker_re_format(ins,rels_out)
            self.process_data.append(pdata)
    def __getitem__(self, item):
        return self.process_data[item]

    def __len__(self):
        return len(self.process_data)
    def convert_to_marker_re_format(self,ins,rels_out):
        lens = len(ins) + 2
        ins = ['[CLS]'] + ins + ['[SEP]']
        pos = [i for i in range(lens)]

        rels = []
        for i,rel_out in enumerate(rels_out):
            rel,(s_entity,s_start,s_end),(o_entity,o_start,o_end) = rel_out
            s_entity_marker_left,s_entity_marker_right = '<s:'+s_entity+'>','</s:'+s_entity+'>'
            o_entity_marker_left,o_entity_marker_right = '<o:' + o_entity + '>', '</o:' + o_entity + '>'
            ins += [s_entity_marker_left,s_entity_marker_right,o_entity_marker_left,o_entity_marker_right]
            pos += [s_start,s_end,o_start,o_end]
            rels.append(rel)
        rel_pos = [lens,len(rels)]
        # attention mask 的构造

        ins_id = self.tokenizer.convert_tokens_to_ids(ins)
        ins_tensor = torch.LongTensor(ins_id)

        pos_en = torch.LongTensor(pos)

        rels_pos = torch.LongTensor(rel_pos)
        rels = torch.LongTensor([self.rel2num[r] if r in self.rel2num else 0 for r in rels])

        padding_mask = torch.ByteTensor([1]*(lens+ 4 * len(rels)))
        # attention_mask 的构造

        attention_mask = self.get_attention_mask(lens, len(rels))
        token_type_id = torch.LongTensor([0]*lens+[1]*(4*len(rels)))
        return ins_tensor,rels,rels_pos,pos_en,padding_mask,token_type_id,attention_mask

    def _collate_fn(self,batch):
        batch = list(zip(*batch))
        raw_attention_masks = batch[-1]
        batch = [pad_sequence(x,batch_first=True,padding_value=0) for x in batch[:-1]]
        max_len = batch[0].size(-1)
        attention_masks = []
        for am in raw_attention_masks:
            tmp = torch.zeros((max_len,max_len))
            m = am.size(0)
            tmp[:m,:m] = am
            attention_masks.append(tmp.bool())
        attention_masks = torch.stack(attention_masks,dim=0)
        return batch + [attention_masks]

    def get_dataloader(self,batch_size,num_workers=0,shuffle=False,pin_memory=False):
        return DataLoader(self,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,
                          collate_fn=self._collate_fn,pin_memory=pin_memory)
    @staticmethod
    def get_attention_mask(lens,marker_nums):
        """
           ins... 1     0     0    0    0    0    0    0
           m1_1   1     1     1    1    1    0    0    0
           m1_2   1
           m1_3   1
           m1-4   1
           m2_1   1
           m2_2   1
           m2_3   1
           m2_4  ins... m1_1 m1_2 m1_3 m1_4 m2_1 m2_2 m2_3 m2_4

           """
        attention_mask = torch.zeros((lens+ 4 * marker_nums,lens+ 4 * marker_nums))
        attention_mask[:,:lens] = 1
        for i in range(marker_nums):
            start,end = lens + 4*i, lens + 4*i + 4
            attention_mask[start:end,start:end] = 1
        return attention_mask.bool()
if __name__ == '__main__':
    w = marker_re_dataset.get_attention_mask(2,2)
    pprint(w)
