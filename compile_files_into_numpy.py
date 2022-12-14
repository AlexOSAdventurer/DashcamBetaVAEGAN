import numpy
from PIL import Image
import os
import os.path
import joblib
import sys

name = sys.argv[1]
output_memmap = name + ".npy" #"train.npy"
folder = name + "/" #"train/"
slice_size = 1000
jobs = 70
paths = [os.path.join(folder, e) for e in os.listdir(folder) if "jpg" in e]
image_count = len(paths)
image_slices = [slice(start, start + slice_size) for start in range(0, image_count - slice_size, slice_size)]
print(name, output_memmap, folder)
def exportToNumpy(memmap_array, our_paths, sl):
    start = sl.start
    stop = sl.stop
    for i in range(start, stop):
        memmap_array[i] = numpy.array(Image.open(our_paths[i - start]), dtype=numpy.uint8).transpose((2,0,1))
        print(i, i - start)

def runItAll():
    output = numpy.lib.format.open_memmap(output_memmap, dtype=numpy.uint8, shape=(image_count, 3, 720, 1280), mode='w+')
    joblib.Parallel(n_jobs=jobs)(joblib.delayed(exportToNumpy)(output, paths[sl], sl)
                   for sl in image_slices)

runItAll()
