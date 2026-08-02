from PIL import Image
import os

path="calibration_set"

for f in os.listdir(path):
    img=Image.open(os.path.join(path,f))
    print(f,img.size,img.mode)