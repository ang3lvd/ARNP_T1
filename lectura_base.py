from os import listdir
from os.path import splitext
import numpy as np


def sub_dir(dir_p, dir_x):
    print(dir_x)
    [busca_jpg(dir_p + dir_j + '/', dir_j) for dir_j in listdir(dir_p)]


def busca_jpg(ddir_p, ddir_x):
    global cont
    arch_jpg = [arch for arch in listdir(ddir_p) if splitext(arch)[1] == '.jpg']
    print('-', ddir_x, ': ', len(arch_jpg))
    cont += len(arch_jpg)



cont = 0
mi_path = "/media/angel/Elements/UFF/imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/"
[sub_dir(mi_path + dir_i + '/', dir_i) for dir_i in listdir(mi_path)]
print('Total de ficheros: ', cont)

