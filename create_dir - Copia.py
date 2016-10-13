from os import listdir
from os.path import splitext
import os
import shutil

def crear_dir(ruta):
    global mi_path, dir_set
    nuevaruta = mi_path + dir_set[1] + ruta + '/'
    if not os.path.exists(nuevaruta):
        os.makedirs(nuevaruta)

def mover_fichero(origen, ddir_x):
    global mi_path, dir_set
    #moviendo para validacion el 25%
    destino =  mi_path + dir_set[1] + ddir_x + '/'
    arch_jpg = [arch for arch in listdir(origen)]
    cant_jpg = len(arch_jpg)
    [shutil.move(origen + arch_jpg[i], destino + arch_jpg[i]) for i in range(int(cant_jpg*0.25)) if os.path.exists(origen + arch_jpg[i])]


def sub_dir(dir_x):
    global mi_path, dir_set
    global_path = mi_path + dir_set[0] + dir_x + '/'
    crear_dir(dir_x)
    mover_fichero(global_path, dir_x)


mi_path = 'G:/UFF/imagesPlaces205_resize/'
dir_set = ['temp_dataset/', 'validation_data/', 'train_data/']
[sub_dir(dir_i) for dir_i in listdir(mi_path + dir_set[0])]


