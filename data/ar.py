from shutil import copy2
import os
for file in os.listdir("arwarp/"):
    if file.endswith(".bmp"):
        gender, idx, features = file.strip(".bmp").split("-")
        print(gender, idx, features)
        # copyfile()
        print(file)
        if not os.path.exists("ardata/" + gender +str(idx)):
            os.makedirs("ardata/" + gender+str(idx))
        if features in ["01", "08", "11"]:
            copy2( "arwarp/" + file,"ardata/" +gender + str(idx) + "/" + str(features) + ".bmp")