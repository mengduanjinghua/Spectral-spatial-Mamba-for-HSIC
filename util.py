import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn import metrics


ce_loss = nn.CrossEntropyLoss()

def tr_acc(model, image, label):
    train_dataset = TensorDataset(torch.tensor(image), torch.tensor(label))
    train_loader = DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
    correct_num = 0
    train_loss = 0
    device = next(model.parameters()).device
    with torch.no_grad():
      for ind, (image_batch, label_batch) in enumerate(train_loader):
          image_batch = image_batch.to(device)
          label_batch = label_batch.to(device)
          pred_array = model(image_batch)
          loss = ce_loss(pred_array, label_batch.long())
          prob, idx = torch.max(pred_array, dim=1)
          train_loss = train_loss + loss.cpu().data.numpy()*image_batch.shape[0]
          correct_num = correct_num + torch.eq(idx, label_batch).float().sum().cpu().numpy()
    return correct_num / image.shape[0], train_loss/image.shape[0]

def test_batch(model, image, index, BATCH_SIZE,  nTrain_perClass, nvalid_perClass, halfsize):
    device = next(model.parameters()).device
    ind = index[0][nTrain_perClass[0]+ nvalid_perClass[0]:,:]
    nclass = len(index)
    true_label = np.zeros(ind.shape[0], dtype = np.int32)
    for i in range(1, nclass):
        ddd = index[i][nTrain_perClass[i] + nvalid_perClass[i]:,:]
        ind = np.concatenate((ind, ddd), axis = 0)
        tr_label = np.ones(ddd.shape[0], dtype = np.int32) * i
        true_label = np.concatenate((true_label, tr_label), axis = 0)
    length = ind.shape[0]
    if length % BATCH_SIZE != 0:
        add_num = BATCH_SIZE - length % BATCH_SIZE
        ff = range(length)
        add_ind = np.random.choice(ff, add_num, replace = False)
        add_ind = ind[add_ind]
        ind = np.concatenate((ind,add_ind), axis =0)
    test_label = np.zeros(ind.shape[0], dtype = np.int32)
    n = ind.shape[0] // BATCH_SIZE
    windowsize = 2 * halfsize + 1
    image_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image.shape[2]], dtype=np.float32)
    for i in range(n):
        for j in range(BATCH_SIZE):
            m = ind[BATCH_SIZE*i+j, :]
            image_batch[j,:,:,:] = image[(m[0] - halfsize):(m[0] + halfsize + 1),
                                                   (m[1] - halfsize):(m[1] + halfsize + 1),:]
        image_b = np.transpose(image_batch,(0,3,1,2))
        pred_array = model(torch.tensor(image_b).to(device))
        pred_label = torch.max(pred_array, dim=1)[1].cpu().data.numpy()
        test_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = pred_label
    predict_label = test_label[range(length)]

    confusion_matrix = metrics.confusion_matrix(true_label, predict_label)
    overall_accuracy = metrics.accuracy_score(true_label, predict_label)

    true_cla = np.zeros(nclass,  dtype=np.int64)
    for i in range(nclass):
        true_cla[i] = confusion_matrix[i,i]
    test_num_class = np.sum(confusion_matrix,1)
    test_num = np.sum(test_num_class)
    num1 = np.sum(confusion_matrix,0)
    po = overall_accuracy
    pe = np.sum(test_num_class*num1)/(test_num*test_num)
    kappa = (po-pe)/(1-pe)*100
    true_cla = np.true_divide(true_cla,test_num_class)*100
    average_accuracy = np.average(true_cla)
    print('overall_accuracy: {0:f}'.format(overall_accuracy*100))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:{0:f}'.format(kappa))
    return true_cla, overall_accuracy*100, average_accuracy, kappa, confusion_matrix, predict_label

def spiral_scan_index(images):#### input: (B, C, H, W)   output: (B, C, H*W)
    height, width, channels = images.size()
    output = torch.zeros([height * width, channels],dtype=torch.long)
    direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = torch.zeros(height, width)

    x, y = 0, 0
    dx, dy = 0, 1

    for i in range(height * width):
        output[i] = images[x, y]
        visited[x, y] = 1
        next_x, next_y = x + dx, y + dy

        if 0 <= next_x < height and 0 <= next_y < width and visited[next_x, next_y] == 0:
            x, y = next_x, next_y
        else:
            dx, dy = direction[(direction.index((dx, dy)) + 1) % 4]
            x, y = x + dx, y + dy

    return output

def generate_matrix(H, W):
    # 生成坐标网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    A = torch.stack((grid_y, grid_x), dim=2)
    return A

def spiral_flatten(images):#### input: (B, C, H, W)   output: (B, C, H*W)
    B, C, H, W = images.size()
    index = generate_matrix(H, W)
    index = spiral_scan_index(index)

    indices = index[:, 0].view(-1, 1, 1), index[:, 1].view(-1, 1, 1)
    image_list = images[:, :, indices[0], indices[1]]
    image_list = image_list.squeeze(-1).squeeze(-1)
    return image_list

def s_flatten(tensor):
    B, C, H, W = tensor.size()
    reshaped_tensor = tensor.view(B, C, -1)

    for i in range(C):
        if i % 2 != 0:
            reshaped_tensor[:, i] = torch.flip(reshaped_tensor[:, i], [1])

    return reshaped_tensor

def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, cm, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')

    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.mean(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.mean(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)

    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements accuracy: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements accuracy: " + str(element_std) + '\n'
    f.write(sentence9)

    f.write("Mean of confusion matrix: " +'\n')
    cm = np.array(cm)
    mean_cm = np.mean(cm, axis = 0)
    for i in range(mean_cm.shape[0]):
        f.write(str(mean_cm[i]) + '\n')
    f.write("########################################################################################################" +'\n'+ '\n')
    f.close()
