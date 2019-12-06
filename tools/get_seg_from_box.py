import json, os


def get_seg_info(json_data, each_box):
    '''
    根据边框得到mask的polygon的点坐标
    '''
    image_width = json_data['image_width']
    image_height = json_data['image_height']
    w = (each_box['x_max'] - each_box['x_min']) * image_width
    h = (each_box['y_max'] - each_box['y_min']) * image_height
    bbox = [each_box['x_min'] * image_width, each_box['y_min'] * image_height, w, h]
    seg = []
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
    return seg


def convert_box_to_seg(json_file_path, label_dir, ori_label_dir):
    with open(os.path.join(ori_label_dir, json_file_path)) as fr:
        json_data = json.load(fr)
    for i in range(json_data['num_box']):
        each_box = json_data['bboxes'][i]
        each_seg = get_seg_info(json_data, each_box)
        all_points_x = each_seg[::2]
        all_points_y = each_seg[1::2]
        mask = {}
        mask['all_points_x'] = all_points_x
        mask['all_points_y'] = all_points_y
        json_data['bboxes'][i]['mask'] = mask

    with open('%s/%s' % (label_dir, json_file_path), 'w') as fr:
        json.dump(json_data, fr)


def main(ori_label_dir, new_label_dir):
    json_list = os.listdir(ori_label_dir)
    for each_json_file in json_list:
        convert_box_to_seg(each_json_file, new_label_dir, ori_label_dir)


if __name__ == '__main__':
    main('/input0/labels', 'labels')
