# this script remove those fiels from casia landmarks which are not present in the masked one

import zipfile
import csv
zfile = zipfile.ZipFile("../masked_casia.zip")
writeFile = open("masked_casia_landmark.txt", "w")
idx = 0
with open("casia_landmark.txt") as f:
    lines = f.readlines()
    # print(lines[0].split("\t"))
    for line in lines:
        idx += 1
        if idx % 5000 == 0:
            print(idx)
            break
        filename = line.split("\t")[0]
        try:
            zfile.open(filename)
            # writeFile.write("")
            print >>writeFile, line
        except:
            print(filename)
            # writeFile.write(filename + "")
writeFile.close()