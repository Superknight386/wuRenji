import numpy as np

train_joint=np.load('/root/wuRenji-main/data/train_joint.npy')#把测试集A的数据加入训练中，用于B测试集测试。
train_label=np.load(('/root/wuRenji-main/data/train_label.npy'))
test_A_joint=np.load('/root/wuRenji-main/data/test_A_joint.npy')
test_A_label=np.load('/root/wuRenji-main/data/test_A_label.npy')
train_bone=np.load('/root/wuRenji-main/data/train_bone.npy')#把测试集A的数据加入训练中，用于B测试集测试。
test_A_bone=np.load('/root/wuRenji-main/data/test_A_bone.npy')
train_label=np.load(('/root/wuRenji-main/data/train_label.npy'))
test_A_joint=np.load('/root/wuRenji-main/data/test_A_joint.npy')
test_A_label=np.load('/root/wuRenji-main/data/test_A_label.npy')
print(train_joint.shape)
print(test_A_joint.shape)
mix_joint=np.concatenate([train_joint,test_A_joint],axis=0)
mix_label=np.concatenate([train_label,test_A_label],axis=0)
print(mix_joint.shape)
print(mix_label.shape)
np.save('/root/wuRenji-main/data/mix_joint.npy',mix_joint)
np.save('/root/wuRenji-main/data/mix_label.npy',mix_label)
np.save('/root/wuRenji-main/data/mix_bone.npy',mix_label)