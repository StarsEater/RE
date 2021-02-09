from prettyprinter import cpprint
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score,f1_score
from functools import reduce
def one_label_f1_score(true_label_lst,pre_label_lst,target_name,way="micro"):
    assert way in ["macro","micro"]
    assert len(true_label_lst) == len(pre_label_lst), \
        print("长度不一致", len(true_label_lst), len(pre_label_lst))
    # print(true_label_lst)
    # print(pre_label_lst)
    print(classification_report(true_label_lst,pre_label_lst,labels=range(len(target_name)),target_names=target_name))
    acc = accuracy_score(true_label_lst,pre_label_lst)
    pre,rec,f1 = precision_score(true_label_lst,pre_label_lst,average=way),\
                 recall_score(true_label_lst,pre_label_lst,average=way),\
                 f1_score(true_label_lst,pre_label_lst,average=way)
    return acc,pre,rec,f1