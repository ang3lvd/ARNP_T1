import numpy as np
import matplotlib.pyplot as plt
import caffe
from os import listdir
from os.path import splitext


def initialize_net():
    # Set the computation mode CPU
    caffe.set_mode_cpu()

    # load the model
    net = caffe.Net('/media/angel/LENOVO/UFF/ARNP/Codes/20161107-182323-69e7_epoch_5.5/deploy.prototxt',
                    '/media/angel/LENOVO/UFF/ARNP/Codes/20161107-182323-69e7_epoch_5.5/snapshot_iter_267663.caffemodel',
                    caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(
        '/media/angel/LENOVO/UFF/ARNP/Codes/20161107-182323-69e7_epoch_5.5/alexnet_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(1, 3, 227, 227)
    return net,transformer

def classify_im(net, transformer, im_path):
    # load the image in the data layer
    im = caffe.io.load_image(im_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    # compute
    out = net.forward()

    #predicted
    return  out['softmax'].flatten()

def busca_jpg(dir_p, dir_x):
    global net, transformer, mtrz_cfsn
    print(dir_x)
    arr = 0
    cont = 0
    for img in listdir(dir_p):
        if splitext(img)[1] == '.jpg':
            arr = arr + classify_im(net, transformer, dir_p + '/' + img)
            cont = cont + 1
    arr = arr/cont
    return arr




net,transformer = initialize_net()
test_path = '/media/angel/LENOVO/UFF/ARNP/Codes/Images/test_data/'
mtrz_cfsn = np.zeros((12,12))
for i,dir_i in enumerate(listdir(test_path)):
    mtrz_cfsn[i,:] = busca_jpg(test_path + dir_i + '/', dir_i)

print(np.round(mtrz_cfsn, 3))





# plt.imshow(im)
# plt.show()





