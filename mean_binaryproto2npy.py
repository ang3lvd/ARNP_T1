import caffe
import numpy as np
import sys


blob = caffe.proto.caffe_pb2.BlobProto()
data = open('/media/angel/LENOVO/UFF/ARNP/Codes/20161107-182323-69e7_epoch_5.5/mean.binaryproto','rb').read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save('/media/angel/LENOVO/UFF/ARNP/Codes/20161107-182323-69e7_epoch_5.5/alexnet_mean.npy', out)