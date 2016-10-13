import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_mode_cpu()
# net = caffe.Net('/media/angel/LENOVO/UFF/ARNP/Codes/conv.prototxt', caffe.TEST)
#
# print(net.inputs)
# [(k, v.data.shape) for k, v in net.blobs.items()]
# [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]
#
# im = np.array(Image.open('/home/angel/Descargas/caffe-master/examples/images/cat_gray.jpg'))
# im_input = im[np.newaxis, np.newaxis, :, :]
# net.blobs['data'].reshape(*im_input.shape)
# net.blobs['data'].data[...] = im_input
#
# net.forward()
#
# for i in range(3):
#     plt.subplot(3,1,i+1)
#     plt.imshow(net.blobs['conv'].data[0,i])
#
# plt.show()

#load the model
net = caffe.Net('/home/angel/Descargas/caffe-master/models/bvlc_alexnet/deploy.prototxt',
                '/home/angel/Descargas/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)
# # load input and configure preprocessing
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_mean('data', np.load('/home/angel/Descargas/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data', 255.0)
im = np.array(Image.open('/home/angel/Descargas/caffe-master/examples/images/cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

#load the image in the data layer
#im = np.array(Image.open('/home/angel/Descargas/caffe-master/examples/images/cat.jpg'))
#net.blobs['data'].data[...] = transformer.preprocess('data', im)

#compute
out = net.forward()

#predicted predicted class
#print( out['prob'].argmax())

#print predicted labels
# labels = np.loadtxt("/home/angel/Descargas/caffe-master/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
# top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
# print(labels[top_k])

print(net.blobs['conv1'].data.shape)
for i in range(net.blobs['conv1'].data.shape[1]):
    plt.subplot(12, 8, i + 1)
    plt.imshow(net.blobs['conv1'].data[0,i,:,:])
    plt.axis('off')
plt.show()