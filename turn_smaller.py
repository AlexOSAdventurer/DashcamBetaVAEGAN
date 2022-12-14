import numpy
from PIL import Image
import cv2
import os
import os.path
import joblib
import sys

input_memmap = numpy.load(sys.argv[1], mmap_mode='r')
slice_size = 1000
jobs = 70
smaller_amount = 128
image_count = input_memmap.shape[0]
output_memmap = numpy.lib.format.open_memmap(sys.argv[2], dtype=numpy.float, shape=(input_memmap.shape[0], input_memmap.shape[1], smaller_amount, smaller_amount), mode='w+')
image_slices = [slice(start, start + slice_size) for start in range(0, image_count - slice_size, slice_size)]
print(sys.argv[1], sys.argv[2])

def transform(input_memmap, output_memmap, sl):
    start = sl.start
    stop = sl.stop
    for i in range(start, stop):
        img = numpy.array(input_memmap[i], dtype=numpy.float)
        output_memmap[i] = (cv2.resize(img.transpose((1,2,0)), dsize=(smaller_amount, smaller_amount), interpolation=cv2.INTER_CUBIC)).transpose((2,0,1))
        if (i % 10 == 0):
            print(i, i - start)

def runItAll():
    joblib.Parallel(n_jobs=jobs)(joblib.delayed(transform)(input_memmap, output_memmap, sl)
                   for sl in image_slices)

runItAll()
