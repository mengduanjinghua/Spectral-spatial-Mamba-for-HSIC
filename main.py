import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


from config import load_args
from data_read import readdata
from generate_pic import generate
from util import tr_acc, test_batch, record_output

from model import mamba_1D_model, mamba_2D_model, mamba_SS_model



day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
args = load_args()

num_of_ex = 10
dataset = args.dataset
windowsize = args.windowsize
type = args.type


train_num = args.train_num

lr = args.lr
epoch = args.epoch
batch_size = args.batch_size
drop_rate = args.drop_rate
gamma = args.lr_decay

model_id = args.model_id

spe_windowsize = args.spe_windowsize
spa_patch_size = args.spa_patch_size
spe_patch_size = args.spe_patch_size
embed_dim = args.embed_dim
hid_chans = args.hid_chans
depth = args.depth
use_bi = args.use_bi
use_global = args.use_global
use_cls = args.use_cls

halfsize = int((windowsize-1)/2)
train_image, train_label, validation_image, validation_label,nTrain_perClass, nvalid_perClass, index,image, gt,s = readdata(args, 0)
gt = gt.astype(np.int32)
nclass = np.max(gt)
result = np.zeros([nclass+3, num_of_ex])
nband = train_image.shape[-1]


net_name_candidate = ['mamba_1D_model', 'mamba_2D_model', 'mamba_SS_model']
net_name = net_name_candidate[model_id]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ce_loss = torch.nn.CrossEntropyLoss()
AC, OA, AA, KA, CM,  TRAINING_TIME, TESTING_TIME = [], [], [], [], [], [], []

for num in range(0, num_of_ex):
    print('num:', num)
    train_image, train_label, validation_image, validation_label,nTrain_perClass, nvalid_perClass, index,image, gt,s = readdata(args, num)
    train_image = np.transpose(train_image,(0,3,1,2))
    validation_image = np.transpose(validation_image,(0,3,1,2))
    nvalid_perClass = np.zeros_like(nvalid_perClass)
    

    if model_id == 0:
        model = mamba_1D_model(img_size=(spe_windowsize,spe_windowsize), spa_img_size=(windowsize, windowsize), nband=nband, patch_size=spe_patch_size, embed_dim=embed_dim, nclass=nclass, depth=depth, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls)
    elif model_id == 1:
        model = mamba_2D_model(img_size=(windowsize, windowsize), patch_size=spa_patch_size, in_chans=nband, hid_chans = hid_chans, embed_dim=embed_dim, nclass=nclass, drop_path=drop_rate, depth=4, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls)
    elif model_id == 2:  
        model = mamba_SS_model(spa_img_size=(windowsize, windowsize),spe_img_size=(spe_windowsize,spe_windowsize), spa_patch_size=spa_patch_size, spe_patch_size=spe_patch_size, in_chans=nband, hid_chans = hid_chans, embed_dim=embed_dim, drop_path=drop_rate, nclass=nclass, depth=depth, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls, fu = args.use_fu)
    else:
        raise Exception('model id does not find')
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr = lr, weight_decay = 1e-4)
    print('the number of training samples:', train_image.shape[0])

    train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_label))
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True, drop_last=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [80, 140, 170], gamma = gamma, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=5,T_mult=2)
    # training
    tic1 = time.time()
    for i in range(epoch):
        model.train()
        train_loss = 0
        for idx, (label_x, label_y) in enumerate(train_loader):
            label_x, label_y = label_x.to(device), label_y.to(device)

            outputs = model(label_x)
            loss = ce_loss(outputs, label_y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()*label_x.shape[0]

        train_loss = train_loss/train_image.shape[0]
        scheduler.step()
        
        if (i+1) % 10 == 0:
            train_acc, train_loss = tr_acc(model.eval(), train_image, train_label)
            val_acc, val_loss = tr_acc(model.eval(), validation_image, validation_label)
            
            print('epoch:', i, 'loss:%.4f' % train_loss,'train_acc:%.4f'%train_acc.item(), 'val_acc:%.4f'%val_acc.item())
    toc1 = time.time()

    true_cla, overall_accuracy, average_accuracy, kappa, cm, test_pred= test_batch(model.eval(), image, index, 400,  nTrain_perClass, nvalid_perClass, halfsize)
    toc2 = time.time()
    
    classification_map, gt_map = generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, overall_accuracy, halfsize, dataset, day_str, num, net_name)
    result[:nclass,num] = true_cla
    result[nclass,num] = overall_accuracy
    result[nclass+1,num] = average_accuracy
    result[nclass+2,num] = kappa
    
    AC.append(true_cla)
    OA.append(overall_accuracy)
    AA.append(average_accuracy)
    KA.append(kappa)
    CM.append(cm)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - toc1)
    ELEMENT_ACC = np.array(AC)
    
np.save('./record/'+ net_name +'_'+ day_str + '_' +dataset+'_'+str(train_image.shape[0]) + '_epoch_' + str(epoch) + '_spa_patch_size_' + str(spa_patch_size) +'_spe_patch_size_'+str(spe_patch_size) + '_embed_dim_'+str(embed_dim)+'_depth _'+str(depth)+ '_use_global_'+str(use_global) +'_use_bi _'+str(use_bi)+'_hdi_chans_'+str(hid_chans)+'_lr _'+str(lr)+'_lrdecay_'+str(gamma)+ '_fusion_'+str(args.use_fu) +'_.npy', result)
    
record_output(OA, AA, KA, ELEMENT_ACC, CM, TRAINING_TIME, TESTING_TIME, './record/' + net_name +'_'+ day_str + '_' +dataset+'_'+str(train_image.shape[0])+ '_epoch_' + str(epoch)+ '_spa_patch_size_' +str(spa_patch_size) +'_spe_patch_size_'+str(spe_patch_size) + '_embed_dim_'+str(embed_dim)+'_depth _'+str(depth)+ '_use_global_'+str(use_global) +'_use_bi _'+str(use_bi)+'_hdi_chans_'+str(hid_chans)+'_lr _'+str(lr)+'_lrdecay_'+str(gamma)+ '_fusion_'+str(args.use_fu) + '_end.txt') 

jingdu = np.mean(result, axis = -1)
print(jingdu)