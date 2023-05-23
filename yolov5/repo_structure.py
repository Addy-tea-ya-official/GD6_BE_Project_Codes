import shutil
import os, sys, random
from lxml import etree as et
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

annotations = sorted(glob('D:\\dataset\\xml_704_480\\*.xml'))
 
df = []
cnt = 0
for file in annotations:
  prev_filename = '.'.join(file.split('\\')[-1].split('.')[:-1])+'.jpeg'
  filename = str(cnt) + '.jpeg'
  row = []
  #parsedXML = ET.parse(file)
  root = et.parse(file).getroot()
  width =  int(float(root[3][0].text))
  height = int(float(root[3][1].text))
  # width = 704
  # height = 576
  for i in range(len(root.xpath(".//object"))):   
      class_name = root[5+i][0].text
      if(class_name == "vehicle"):
        #print(root[3].text)
        xmin = int(float(root[5+i][2][0].text))
        xmax = int(float(root[5+i][2][2].text))
        ymin = int(float(root[5+i][2][1].text))
        ymax = int(float(root[5+i][2][3].text))
        row = [prev_filename, filename, class_name, xmin, xmax, ymin, ymax,width,height]
        # print(row)
        df.append(row)
      # print(row)
  #print(filename)

  '''
  for node in parsedXML.getroot().iter('ball'):
      blood_cells = node.find('name').text
      xmin = int(node.find('bndbox/xmin').text)
      xmax = int(node.find('bndbox/xmax').text)
      ymin = int(node.find('bndbox/ymin').text)
      ymax = int(node.find('bndbox/ymax').text)
      row = [prev_filename, filename, blood_cells, xmin, xmax, ymin, ymax]
      df.append(row)
  '''
  '''
  for node in parsedXML.getroot().iter('tail'):   
      blood_cells = node.find('name').text
      xmin = int(node.find('bndbox/xmin').text)
      xmax = int(node.find('bndbox/xmax').text)
      ymin = int(node.find('bndbox/ymin').text)
      ymax = int(node.find('bndbox/ymax').text)
      row = [prev_filename, filename, blood_cells, xmin, xmax, ymin, ymax]
      df.append(row)
  '''   
  cnt += 1
 
data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax','img_width','img_height'])

def width(df):
  return int(df.xmax - df.xmin)
def height(df):
  return int(df.ymax - df.ymin)
def x_center(df):
  return int(df.xmin + (df.width/2))
def y_center(df):
  return int(df.ymin + (df.height/2))

le = preprocessing.LabelEncoder()
le.fit(data['cell_type'])
print(le.classes_)
labels = le.transform(data['cell_type'])
data['labels'] = labels

data['width'] = data.apply(width, axis=1)
data['height'] = data.apply(height, axis=1)

data['x_center'] = data.apply(x_center, axis=1)
data['y_center'] = data.apply(y_center, axis=1)

def w_norm(df):
  return df.x_center/df.img_width

def w_norm1(df):
  return df.width/df.img_width

def h_norm(df):
  return df.y_center/df.img_height

def h_norm1(df):
  return df.height/df.img_height
  
data['x_center_norm'] = data.apply(w_norm, axis=1)

data['width_norm'] = data.apply(w_norm1, axis=1)

data['y_center_norm'] = data.apply(h_norm, axis=1)
data['height_norm'] = data.apply(h_norm1, axis = 1)

df_train, df_valid = model_selection.train_test_split(data, test_size=0.25, random_state=13, shuffle=False)
try:
    os.mkdir('D:\\dataset\\yolov5\\bcc1\\')
    os.mkdir('D:\\dataset\\yolov5\\bcc1\\images\\')
    os.mkdir('D:\\dataset\\yolov5\\bcc1\\images\\train\\')
    os.mkdir('D:\\dataset\\yolov5\\bcc1\\images\\valid\\')

    os.mkdir('D:\\dataset\\yolov5\\bcc1\\labels\\')
    os.mkdir('D:\\dataset\\yolov5\\bcc1\\labels\\train\\')
    os.mkdir('D:\\dataset\\yolov5\\bcc1\\labels\\valid\\')
except:
    pass

def segregate_data(df, img_path, label_path, train_img_path, train_label_path):
  filenames = []
  for filename in df.filename:
    filenames.append(filename)
  filenames = set(filenames)
  
  for filename in filenames:
    yolo_list = []

    for _,row in df[df.filename == filename].iterrows():
      yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

    yolo_list = np.array(yolo_list)

    
    #print(os.path.join(img_path,row.prev_filename), os.path.join(train_img_path,row.prev_filename))
    
    shutil.copyfile(os.path.join(img_path,row.prev_filename), os.path.join(train_img_path,row.prev_filename))
    
    #print('.'.join(row.prev_filename.split('.')[:-1]))
    txt_filename = os.path.join(train_label_path,str('.'.join(row.prev_filename.split('.')[:-1]))+".txt")
    #print(txt_filename)
    # Save the .img & .txt files to the corresponding train and validation folders
    np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
  

src_img_path = "D:\\dataset\\img_704_480"
src_label_path = "D:\\dataset\\xml_704_480"

train_img_path = "D:\\dataset\\yolov5\\bcc1\\images\\train"
train_label_path = "D:\\dataset\\yolov5\\bcc1\\labels\\train"

valid_img_path = "D:\\dataset\\yolov5\\bcc1\\images\\valid"
valid_label_path = "D:\\dataset\\yolov5\\bcc1\\labels\\valid"
print('.')
segregate_data(df_train, src_img_path, src_label_path, train_img_path, train_label_path)
segregate_data(df_valid, src_img_path, src_label_path, valid_img_path, valid_label_path)
