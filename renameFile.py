import os

name =  "gaeul"
path = f"./resource/video/{name}_face/"
A = os.listdir(path)
print("number of object: ", len(A))

i = 0
for i in range(len(A)):
    old_name = path+A[i]
    new_name = path + name + str(i) + '.jpg'
    try:
        os.rename(old_name, new_name)
    except FileExistsError:
        print("File already exists!")
    except:
        print("Something wrong! Check it again.")