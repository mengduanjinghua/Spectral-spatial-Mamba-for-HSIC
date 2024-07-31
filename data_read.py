import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

def pca_whitening(image, number_of_pc):

    shape = image.shape

    image = np.reshape(image, [shape[0]*shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components = number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc),dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))
    return pc_images

def load_data(dataset):
    ###下面是讲解python怎么读取.mat文件以及怎么处理得到的结果###
    data_dir = '/home/hlb/dataset'
    if dataset == 'Indian':
        image_file = data_dir + '/Indian/Indian_pines_corrected.mat'
        label_file = data_dir + '/Indian/Indian_pines_gt.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['indian_pines_corrected']
        label = label_data['indian_pines_gt']
    elif dataset == 'Pavia':
        image_file = data_dir + '/Pavia/Pavia.mat'
        label_file = data_dir + '/Pavia/Pavia_groundtruth.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['paviaU']#pavia1
        label = label_data['paviaU_gt']#pavia1
    elif dataset == 'Houston':
        image_file = data_dir + '/Houston/CASI.mat'
        label_file = data_dir + '/Houston/CASI_gnd_flag.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['CASI']
        label = label_data['gnd_flag']  # houston
    else:
        raise Exception('dataset does not find')
    image = image.astype(np.float32)
    return image, label

def readdata(args, num):

    or_image, or_label = load_data(args.dataset)
    windowsize = args.windowsize
    train_num = args.train_num
    halfsize = int((windowsize-1)/2)
    number_class = np.max(or_label)
    
    #preprocessing
    image = np.pad(or_image, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    label = np.pad(or_label, ((halfsize, halfsize), (halfsize, halfsize)), 'constant',constant_values=0)
    if args.type == 'PCA':
        image1 = pca_whitening(image, number_of_pc = 30)
    if args.type == 'none':
        image1 = image
    image = image1
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    #set the manner of selecting training samples

    n = np.zeros(number_class,dtype=np.int64)
    for i in range(number_class):
        temprow, tempcol = np.where(label == i + 1)
        n[i] = len(temprow)
    total = np.sum(n)

    ####每类按比例选择训练样本
    nTrain_per = 0.01
    nTrain_perClass = np.ceil(nTrain_per * n)
    nTrain_perClass = nTrain_perClass.astype(np.int32)

    ####每类按固定数量选择训练样本
    nTrain_perClass = np.ones(number_class,dtype=np.int32) * train_num
    ind = np.where(n <= train_num)
    if train_num <=15:
        nTrain_perClass[ind[0]] = train_num//2
    else:
        nTrain_perClass[ind[0]] = 15


    num_vali = 200
    nValidation_perClass =  (n/total)*num_vali
    nvalid_perClass = nValidation_perClass.astype(np.int32)


    index = []
    s = 0
    flag = 0
    f = 0
    bands = np.size(image,2)
    validation_image = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands], dtype=np.float32)
    validation_label = np.zeros(np.sum(nvalid_perClass), dtype=np.int32)
    train_image = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands], dtype=np.float32)
    train_label = np.zeros(np.sum(nTrain_perClass),dtype=np.int32)


    for i in range(number_class):

        temprow, tempcol = np.where(label == i + 1)
        s = s + len(temprow)
        matrix = np.zeros([len(temprow),2], dtype=np.int64)
        matrix[:,0] = temprow
        matrix[:,1] = tempcol
        np.random.seed(num)
        np.random.shuffle(matrix)

        temprow = matrix[:,0]
        tempcol = matrix[:,1]
        index.append(matrix)

        for j in range(nTrain_perClass[i]):
            train_image[flag + j, :, :, :] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                            (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_label[flag + j] = i
        flag = flag + nTrain_perClass[i]

        for j in range(nTrain_perClass[i], nTrain_perClass[i] + nvalid_perClass[i]):
            validation_image[f -nTrain_perClass[i]+ j, :, :, :] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                            (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            validation_label[f-nTrain_perClass[i] + j] = i
        f = f + nvalid_perClass[i]

    return train_image, train_label,validation_image, validation_label, nTrain_perClass, nvalid_perClass, index, image, label, s