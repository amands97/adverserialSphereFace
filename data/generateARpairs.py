import os
import random
# random.sample(the_list, 50)

peopleNames = os.listdir("./ardata/")
for file in peopleNames:
    if file.endswith(".zip"):
        peopleNames.remove(file)
    if file.endswith(".DS_Store"):
        peopleNames.remove(file)
print(peopleNames)
# print(len(peopleNames))




with open("train_ardata_pairs.txt", "w") as writeFile:
    for p1 in peopleNames:
        # file1 = p1 + "/" + "08.bmp"
        # file2 = p1 + "/" + "11.bmp"
        # writeFile.write(p1 + " 01 "  11\n")
        p2list = random.sample(peopleNames, 5)
        for p2 in p2list:
            if p2 == p1:
                continue
            writeFile.write(p1 + " 01 " + p2 + " 01\n")
            # writeFile.write(p1 + " 08 " + p2 + " 11\n")
            # writeFile.write(p1 + " 11 " + p2 + " 08\n")
            # writeFile.write(p1 + " 11 " + p2 + " 11\n")

with open("test_ardata_pairs.txt", "w") as writeFile:
    for p1 in peopleNames:
        # file1 = p1 + "/" + "08.bmp"
        # file2 = p1 + "/" + "11.bmp"
        writeFile.write(p1 + " 08 11\n")
        p2list = random.sample(peopleNames, 5)
        for p2 in p2list:
            if p2 == p1:
                continue
            writeFile.write(p1 + " 08 " + p2 + " 08\n")
            writeFile.write(p1 + " 08 " + p2 + " 11\n")
            writeFile.write(p1 + " 11 " + p2 + " 08\n")
            writeFile.write(p1 + " 11 " + p2 + " 11\n")

