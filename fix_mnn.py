import os
import json

img_dir = os.path.abspath("calibration_set").replace("\\", "/")

valid_imgs = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# 生成图片列表
image_list = []

for img in valid_imgs:
    image_list.append(
        img_dir + "/" + img
    )

with open("images.txt", "w", encoding="utf-8") as f:
    for img in image_list:
        f.write(img + "\n")


config = {
    "format": "RGB",
    "mean": [0.0, 0.0, 0.0],
    "normal": [
        0.0039215,
        0.0039215,
        0.0039215
    ],
    "width": 1920,
    "height": 1080,
    "path": os.path.abspath("images.txt").replace("\\", "/"),
    "used_image_num": len(image_list),
    "feature_quantize_method": "KL",
    "weight_quantize_method": "MAX_ABS"
}


with open("quantconfig.json","w",encoding="utf-8") as f:
    json.dump(config,f,indent=4)


print("图片数量:",len(image_list))
print("images.txt 已生成")