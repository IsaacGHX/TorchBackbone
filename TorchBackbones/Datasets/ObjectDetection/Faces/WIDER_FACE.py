from _utils.read_annots import get_data
from tqdm import tqdm

data_path = "D:/datasets/WIDER_val/WIDER_val/images"
label_path = "D:/datasets/wider_face_split/wider_face_val_bbx_gt.txt"

img_paths, gt_boxes, gt_labels = get_data(data_path, label_path)
# img_paths: the absolute path of datasets' images which include several faces on single image.
# gt_boxes: the format is the (ltx, lty, w, h)
# gt_labels: [blur_level, expression, illumination, invalid, occlusion, pose]
print("Image Path reading finished! www")

print("length of the dataset is: ", len(img_paths))

if __name__ == '__main__':
    import cv2

    # print(gt_labels[0])

    RATIO_THRE = 10 * 10 / 336
    SMALL_RELATIVE_COUNT = 0
    SMALL_ABS_COUNT = 0
    ALL_FACE_NUM = 0
    ONE_FACE_IMG_COUNT = 0
    MUCH_FACE_IMG_COUNT = 0
    INVALID_SMALL_FACE = 0

    for i, img_dir in tqdm(enumerate(img_paths)):

        img = cv2.imread(img_dir)
        h, w, c = img.shape
        img_size = h * w
        ALL_FACE_NUM += len(gt_boxes[i])

        if len(gt_boxes[i]) == 1:
            ONE_FACE_IMG_COUNT += 1
        if len(gt_boxes[i]) >= 5:
            MUCH_FACE_IMG_COUNT += 1

        # print(img.shape)
        # print(len(gt_boxes[i]))
        # print(len(gt_labels[i]))

        for gt_box, gt_label in zip(gt_boxes[i], gt_labels[i]):
            box_area = gt_box[2] * gt_box[3]
            if box_area / img_size <= RATIO_THRE:
                SMALL_RELATIVE_COUNT += 1

            if box_area <= 100:
                SMALL_ABS_COUNT += 1
                if gt_label[3] == 1:
                    INVALID_SMALL_FACE += 1

    print("Total face num is : ", ALL_FACE_NUM)
    print("Total relative small face num is : ", SMALL_RELATIVE_COUNT)
    print("Total abs small face num is : ", SMALL_ABS_COUNT)
    print("Total invalid small face IMG num is : ", INVALID_SMALL_FACE)

    print("Total ONE face IMG num is : ", ONE_FACE_IMG_COUNT)
    print("Total MUCH face IMG num is : ", MUCH_FACE_IMG_COUNT)
