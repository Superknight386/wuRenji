import torch
import pickle
import argparse
import numpy as np
import pandas as pd


def Cal_Score(File, Rate):
    final_score = torch.zeros(2000, 155)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = np.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values).permute(-1,0)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = []
    val_txt = np.load(val_txt_path)
    return val_txt



# Mix_GCN Score File
j_file = '/workspace/TE-GCN-main/TE-GCN-main/work_dir/ctrgcn_joint_eval/epoch1_test_score.npy'
b_file = '/workspace/TE-GCN-main/TE-GCN-main/work_dir/ctrgcn_bone_eval/epoch1_test_score.npy'
jm_file = '/workspace/TE-GCN-main/TE-GCN-main/work_dir/ctrgcn_motion_eval/epoch1_test_score.npy'
td_j_file='/workspace/TE-GCN-main/TE-GCN-main/work_dir/tdgcn_joint_eval/epoch1_test_score.npy'
td_b_file = '/workspace/TE-GCN-main/TE-GCN-main/work_dir/tdgcn_bone_eval/epoch1_test_score.npy'
td_jm_file = '/workspace/TE-GCN-main/TE-GCN-main/work_dir/tdgcn_motion_eval/epoch1_test_score.npy'


val_txt_file = '/workspace/TE-GCN-main/TE-GCN-main/data/test_A_label.npy'

File = [j_file, b_file, jm_file, td_b_file, td_jm_file]    

Numclass = 155
Sample_Num = 6599
Rate = [1, 0.7, 0.3, 1, 0.7,
        0.3] 
final_score = Cal_Score(File, Rate)
true_label = gen_label(val_txt_file)

Acc = Cal_Acc(final_score, true_label)

print('acc:', Acc)