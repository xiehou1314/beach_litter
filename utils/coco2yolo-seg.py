import json
import pandas as pd
import xml.etree.ElementTree as ET
import os, cv2
import numpy as np
import glob
import pycocotools.mask as mask_utils
from skimage import measure

classes = []

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xmlpath, xmlname):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')

        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))

def query_json_to_txt(dir_json, dir_txt):
    classes2id = {}
    num = 0
    jsons = os.listdir(dir_json)
    for i in jsons:
        json_path = os.path.join(dir_json, i)
        with open(json_path, 'r', encoding="utf-8") as f:
            json_data = json.load(f)
            # print(json_data['shapes'])
            for j in json_data['shapes']:
                if j['label'] not in classes2id:
                    classes2id[j['label']] = num
                    num += 1

    def json2txt(path_json, path_txt):  # 可修改生成格式
        with open(path_json, 'r', encoding='utf-8') as path_json:
            jsonx = json.load(path_json)
            with open(path_txt, 'w+') as ftxt:
                shapes = jsonx['shapes']
                # 获取图片长和宽
                width = jsonx['imageWidth']
                height = jsonx['imageHeight']
                # print(shapes)
                cat = shapes[0]['label']
                cat = classes2id[cat]

                for shape in shapes:
                    # 获取矩形框两个角点坐标
                    x1 = shape['points'][0][0]
                    y1 = shape['points'][0][1]
                    x2 = shape['points'][1][0]
                    y2 = shape['points'][1][1]

                    dw = 1. / width
                    dh = 1. / height
                    x = dw * (x1 + x2) / 2
                    y = dh * (y1 + y2) / 2
                    w = dw * abs(x2 - x1)
                    h = dh * abs(y2 - y1)
                    yolo = f"{cat} {x} {y} {w} {h} \n"
                    ftxt.writelines(yolo)

    list_json = os.listdir(dir_json)
    for cnt, json_name in enumerate(list_json):
        if os.path.splitext(json_name)[-1] == ".json":
            path_json = dir_json + json_name
            path_txt = dir_txt + json_name.replace('.json', '.txt')
            json2txt(path_json, path_txt)

    pd.DataFrame([{"原始类别": k, "编码": v} for k, v in classes2id.items()]).to_excel("label_codes.xlsx", index=None)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

###################### 实例分割处理 #######################################
def parse_json_for_instance_segmentation(json_path, label_dict={}):
    # coco格式 annotation的counts中存在顶点集 yolo格式 类别 顶点集
    # 1.创建文件
    # 2.获取类别id
    # 3.转化counts顶点集 存入
    # 4.换行
    # 打开并读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as file:
        json_info = json.load(file)

    annotations = []  # 初始化一个用于存储注释的列表
    # 遍历JSON文件中的每个形状（shape）
    for shape in json_info["annotations"]:
        label = shape["category_id"]  # 获取实例分割类别的标签

        # 获取顶点集
        rle = shape["segmentation"]  # 获取rle

        mask=np.array(mask_utils.decode(rle),dtype=np.float32)

        tolerance=0
        polygons=[]

        padded_binary_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        print("/n")
        print(contours)
        contours = np.subtract(contours, 1)
        for contour in contours:
            contour = close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        # print(segmentation)

        #有类别 顶点集了 然后创建文件 输入 换行

        # print(polygons)

        i=0
        points_str=' '

        for point in segmentation:
            if i%2==0:
                # print(size[0])
                point=round(point/shape['segmentation']['size'][1],6)
                # point=size[0]
            else:
                point=round(point/shape['segmentation']['size'][0],6)
            points_str+=str(point)
            i+=1
            points_str+=' '
            # print(point)

        # print(len(segmentation))

        # print(rle)
        print(points_str)

        # 将点转换为字符串格式，并按照图像的宽度和高度进行归一化
        # points_str = ' '.join(
        #     [f"{round(point / shape['segmentation']['size'][0], 6)} {round(point / json_info['segmentation']['size'][1], 6)}" for point
        #      in segmentation])

        # print(points_str)
        annotation_str = f"{label} {points_str}\n"  # 将编码和点字符串合并为一行注释
        annotations.append(annotation_str)  # 将该注释添加到列表中

        # return 1

    return annotations, label_dict  # 返回注释列表和更新后的标签字典

def process_directory_for_instance_segmentation(json_dir, txt_save_dir, label_dict=None):
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)

    # 如果没有提供初始的label_dict，则从空字典开始
    final_label_dict = label_dict.copy() if label_dict is not None else {}

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        txt_path = os.path.join(txt_save_dir, json_file.replace(".json", ".txt"))

        # 每次解析时传递当前的final_label_dict
        annotations, updated_label_dict = parse_json_for_instance_segmentation(json_path, final_label_dict)

        # 更新final_label_dict，确保包括所有新标签及其编码
        final_label_dict.update(updated_label_dict)

        # 检查annotations是否为空，如果为空则跳过写入操作
        if annotations:
            with open(txt_path, "w") as file:
                file.writelines(annotations)

    # 保存最终的标签字典
    pd.DataFrame(list(final_label_dict.items()), columns=['原始Label', '编码后Label']).to_excel('label_codes.xlsx',
                                                                                           index=False)
    return final_label_dict

# 传入参数为 实例分割 和 目标检测
query_type = "实例分割"
# 传入原始标签数据，比如xml、json格式文件所在的目录下
label_directory = '../datasets/Beach Plastic Litter Dataset version 2/BePLi_dataset_v2/plastic_coco/annotation'
# 填写想转换的txt输出到哪里
output_directory = '../datasets/Beach Plastic Litter Dataset version 2/BePLi_dataset_v2/plastic_coco/labels-seg/'

anno_files = glob.glob(label_directory + "*")
file_type = anno_files[0].split(".")[-1]

label_dict = {
    # 如果想预设标签就在这里填入对应的键值对
}

if query_type == "实例分割":
    process_directory_for_instance_segmentation(label_directory, output_directory, label_dict)

elif query_type == "目标检测":
    if file_type == "json":
        query_json_to_txt(label_directory, output_directory)

    ## 处理xml格式文件
    elif file_type == "xml" or file_type == "XML":
        postfix = 'jpg'
        imgpath = 'query_data/xy/images'
        xmlpath = 'query_data/xy/Annotations'
        txtpath = 'query_data/xy/txt'

        if not os.path.exists(txtpath):
            os.makedirs(txtpath, exist_ok=True)

        file_list = glob.glob(xmlpath + "/*")
        error_file_list = []
        for i in range(0, len(file_list)):
            try:
                path = file_list[i]
                if ('.xml' in path) or ('.XML' in path):
                    convert_annotation(path, path.split("\\")[-1])
                    print(f'file {list[i]} convert success.')
                else:
                    print(f'file {list[i]} is not xml format.')
            except Exception as e:
                print(f'file {list[i]} convert error.')
                print(f'error message:\n{e}')
                error_file_list.append(list[i])
        print(f'this file convert failure\n{error_file_list}')
        print(f'Dataset Classes:{classes}')