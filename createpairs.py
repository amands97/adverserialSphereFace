import os
import csv
import random
# import zipfile
folder = "data/FaceDisguiseDatabase/FaceAll_cropped/"
filenames = os.listdir(folder)
l = lambda x: int(x[3:-5])
subjects = [l(x) for x in filenames]
# zfile.read(name2)
# zfile = zipfile.ZipFile(args.data)

# "sub" + format(i, '03')+ str(ran1) + ".jpg"
for i in range(410):
    print(i)
    # diff person
    ranint = random.randint(0, 409)
    while(ranint == i):
        ranint = random.randint(0, 409)

    # file1 = "sub" + format(i, '03'
    ran1 = random.randint(1, 6)
    file1 = "sub" + format(i, '03')+ str(ran1) + ".jpg"

    while(file1 not in filenames):
        ran1 = random.randint(1, 6)
        file1 = "sub" + format(i, '03')+ str(ran1) + ".jpg"

    ran2 = random.randint(1, 6)
    file2 = "sub" + format(ranint, '03')+ str(ran2) + ".jpg"
    while(file2 not in filenames):
        ran2 = random.randint(1, 6)
        file2 = "sub" + format(ranint, '03')+ str(ran2) + ".jpg"

    with open("pairs.txt", 'a') as f:
        f.write("%s, %s, 0\n"%("sub" + format(i, '03')+ str(ran1) + ".jpg", "sub" +  format(ranint, '03') + str(ran2) + ".jpg"))

    # same person
    ran1 = random.randint(1, 6)
    file1 = "sub" + format(i, '03')+ str(ran1) + ".jpg"
    while(file1 not in filenames):
        ran1 = random.randint(1, 6)
        file1 = "sub" + format(i, '03')+ str(ran1) + ".jpg" 
    
        # ran1 = 
    ran2 = random.randint(1,6)
    file2 = "sub" + format(i, '03')+ str(ran2) + ".jpg"
    while (ran1 == ran2 or file2 not in filenames):
        ran2 = random.randint(1,6)
        file2 = "sub" + format(i, '03')+ str(ran2) + ".jpg"

    
    with open("pairs.txt", 'a') as f:
        f.write("%s, %s, 1\n"%("sub" + format(i, '03')+ str(ran1) + ".jpg", "sub" +  format(i, '03') + str(ran2) + ".jpg"))