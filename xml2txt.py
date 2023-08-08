import xml.etree.ElementTree as ET
import glob
import os
import json
import argparse
import shutil


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

parser = argparse.ArgumentParser(
    description='get input directory')
parser.add_argument('-i','--src', dest='IMG', type= str,
                    help='images folder')
parser.add_argument('-o','--lbl', dest='LBL', type= str,
                    help='annotation folder')
parser.add_argument('-d','--dest', dest='DST', type= str,
                    help='folder that contrain images and labels together')
args = vars(parser.parse_args())


src = args['IMG']
lbl = args['LBL']
dest = args['DST']
# create the dest folder (output directory)
if not os.path.exists(dest):
    os.mkdir(dest)
for path, subdirs, files in os.walk(lbl):
    i=0
    for name in files:
        try:
            # copy only images which has labels
            filename = os.path.join(path, name)
            imgname = os.path.join(src + os.path.basename(path), name[:-3]+'JPEG')
            shutil.copy2(imgname , dest)
            shutil.copy2(filename, dest)
            i+=1
        except FileNotFoundError:
            i-=1

print(f"moved {i} annotations")


classes = []
input_dir = dest
output_dir = dest
image_dir = dest



# identify all the xml files in the annotations folder (input directory)
files = glob.glob(os.path.join(input_dir, '*.xml'))
# loop through each 
for fil in files:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    # check if the label contains the corresponding image file
    if not os.path.exists(os.path.join(image_dir, f"{filename}.JPEG")):
        print(f"{filename} image does not exist!")
        continue

    result = []

    # parse the content of the xml file
    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall('object'):
        label = obj.find("name").text
        # check for new classes and append to list
        if label not in classes:
            classes.append(label)
        index = classes.index(label)
        pil_bbox = [int(x.text) for x in obj.find("bndbox")]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
        # convert data to string
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{index} {bbox_string}")

#### please append the label not the index

    if result:
        # generate a YOLO format text file for each xml file
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result))

# generate the classes file as reference
with open('classes.txt', 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))