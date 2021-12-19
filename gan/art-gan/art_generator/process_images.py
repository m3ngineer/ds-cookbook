# https://github.com/robgon-art/MAGnet/blob/main/2_MAGnet_Process_Modern_Paintings.ipynb

from google.colab import drive
drive.mount('/content/drive')

!mkdir /content/drive/MyDrive/cubist_art_resized_512
!mkdir /content/drive/MyDrive/cubist_art_resized_512/paintings

import glob
image_files = sorted(glob.glob("/content/drive/MyDrive/cubist_paintings*/*.jpg"))
num_files = len(image_files)
print(num_files)

from PIL import Image
import matplotlib.pyplot as plt

def save_image(file, img, side):
  parts = file.split("/")
  path = "/content/drive/MyDrive/cubist_art_resized_512/paintings/" + parts[-1][:-4:] + side + ".jpg"
  # print(path)
  print(str(i).zfill(4), "of", num_files, path)
  img.save(path)

for i, file in enumerate(sorted(image_files)):
  try:
    img = Image.open(file)
    img = img.convert('RGB')
  except:
    continue
  width, height = img.size

  if (width > height): # wide
    # print("wide")
    img1 = img.crop((0, 0, height, height))
    img1 = img1.resize((512, 512))
    save_image(file, img1, "_lft")

    offset = (width-height) // 2
    img2 = img.crop((offset, 0, height+offset, height))
    img2 = img2.resize((512, 512))
    save_image(file, img2, "_ctr")

    img3 = img.crop((width-height, 0, width, height))
    img3 = img3.resize((512, 512))
    save_image(file, img3, "_rgt")

  elif (height > width): # tall
    # print("tall")
    img1 = img.crop((0, 0, width, width))
    img1 = img1.resize((512, 512))
    save_image(file, img1, "_top")

    offset = (height-width) // 2
    img2 = img.crop((0, offset, width, width+offset))
    img2 = img2.resize((512, 512))
    save_image(file, img2, "_ctr")

    img3 = img.crop((0, height-width, width, height))
    img3 = img3.resize((512, 512))
    save_image(file, img3, "_bot")

  else: # square
    # print("square")
    img1 = img
    save_image(file, img1, "")

  # imgplot = plt.imshow(img1)
  # plt.axis("off")
  # plt.show()
  # imgplot = plt.imshow(img2)
  # plt.axis("off")
  # plt.show()
  # imgplot = plt.imshow(img3)
  # plt.axis("off")
  # plt.show()

  # if i > 5:
  #   break
