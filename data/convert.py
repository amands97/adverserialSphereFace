# this script remove those fiels from casia landmarks which are not present in the masked one

import zipfile
import csv
zfile = zipfile.ZipFile("../masked_casia.zip")

with open("casia_landmark.txt") as f:
    lines = f.readlines()
    # print(lines[0].split("\t"))
    for line in lines:
        filename = line.split("\t")[0]
        try:
            zfile.open(filename)
        except:
            print(filename)