import torch
import torch.nn as nn
from transformers import BertModel

from dataset.marker_re.dataset import marker_re_dataset


class marker_re_model(nn.Module):
    def __init__(self,bert_path,label_num,hidden_size=768,mid_size=768):
        super(marker_re_model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # self.hidden2tag = nn.Linear(hidden_size,label_num)
        self.hidden2tag = nn.Sequential(
            nn.Linear(hidden_size*2,mid_size),
            nn.ReLU(inplace=True),
            nn.Linear(mid_size,label_num)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,ins_tensor,rels,rels_pos,pos_en,padding_mask,token_type_ids,attention_mask,test=False):
        nbatch = ins_tensor.size(0)
        start, num = rels_pos[:, 0], rels_pos[:, 1]
        outs = self.bert(input_ids=ins_tensor,
                        attention_mask=padding_mask,
                        token_type_ids=token_type_ids,
                        encoder_attention_mask = attention_mask,
                        position_ids=pos_en)[0]


        marker_vec = []
        label_vec = []
        for b in range(nbatch):
            b_start,b_num = start[b],num[b]
            for i in range(b_num):
                marker_vec.append(torch.cat([outs[b][b_start+i*4],outs[b][b_start + i * 4 + 2]],dim=-1))
                # marker_vec.append(outs[b][b_start + i * 4] - outs[b][b_start + i * 4 + 2])
                label_vec.append(rels[b][i])

        marker_vec = torch.stack(marker_vec,dim=0)
        label_vec = torch.stack(label_vec,dim=0)
        pred0= self.hidden2tag(marker_vec)
        loss  = self.criterion(pred0,label_vec)
        pred = torch.argmax(pred0,dim=-1)

        # 测试阶段输出
        if test:
            test_preds = []
            cur = 0
            pred = pred.detach().tolist()
            for b in range(nbatch):
                b_num = num[b]
                pred_b = pred[cur:cur+b_num]
                test_preds.append(pred_b)
                cur += b_num
            return test_preds



        return loss,pred,label_vec