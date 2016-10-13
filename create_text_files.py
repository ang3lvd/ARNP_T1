from os import listdir
from os.path import splitext
import numpy as np


def sub_dir(dir_p, dir_x):
    print(dir_x)
    [busca_jpg(dir_p + dir_j + '/', dir_j) for dir_j in listdir(dir_p)]


def busca_jpg(ddir_p, ddir_x):
    global cont_f, cont_d, ft, fl
    arch_jpg = [arch for arch in listdir(ddir_p) if splitext(arch)[1] == '.jpg']
    print('-', ddir_x, ': ', len(arch_jpg))
    [ft.write(ddir_p + arch_i + '  ' + str(cont_d)+'\n') for arch_i in listdir(ddir_p)]
    fl.write(ddir_x+'\n')
    cont_f += len(arch_jpg)
    cont_d += 1


cont_f = 0
cont_d = 0
ft = open('/media/angel/Elements/UFF/imagesPlaces205_resize/training_dir.txt', 'w')
fl = open('/media/angel/Elements/UFF/imagesPlaces205_resize/labels.txt', 'w')
mi_path = "/media/angel/Elements/UFF/imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/"
[sub_dir(mi_path + dir_i + '/', dir_i) for dir_i in listdir(mi_path)]
print('Total de ficheros: ', cont_f)
ft.close()
fl.close()


