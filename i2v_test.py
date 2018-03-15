import i2v
from PIL import Image
import os
from os import listdir
from os.path import join
import pickle
import numpy as np
from anime_face_detector import detect
import cv2
import chainer
import shutil
import json

print(chainer.cuda.available)
print(chainer.cuda.cudnn_enabled)

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# print(os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"])

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

# illust2vec2 = i2v.make_i2v_with_chainer("illust2vec_ver200.caffemodel")

# In the case of caffe, please use i2v.make_i2v_with_caffe instead:
# illust2vec = i2v.make_i2v_with_caffe(
#     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
#     "tag_list.json")

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

resize_scale = 256

# n_tags = 35
# tag_info = json.loads(open('tag_info.json', 'r').read())
# tag_list = []
# for tag in list(tag_info[str(n_tags)].values())[2:6]:
#     for v in tag:
#         tag_list.append(v)

dataset = 'getchu_enhanced'
data_root = '../Data'
data_path = join(data_root, dataset, 'train')
# data_path2 = dataset2
# data_path3 = dataset3
# save_path = join(data_root, new_dataset, 'full')
# save_path1 = join(data_root, new_dataset, 'train')
# save_path2 = join(data_root, new_dataset, 'not_girl')
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# if not os.path.exists(save_path1):
#     os.makedirs(save_path1)
# if not os.path.exists(save_path2):
#     os.makedirs(save_path2)

cascade_file = "lbpcascade_animeface.xml"
scale = 1.5
min_size = 80

img_list = [join(data_path, x) for x in sorted(listdir(data_path)) if is_image_file(x)]
# img_list2 = [join(data_path2, x) for x in sorted(listdir(data_path2)) if is_image_file(x)]
# img_list3 = [join(data_path3, x) for x in sorted(listdir(data_path3)) if is_image_file(x)]
# img_list = img_list1 + img_list2 + img_list3
num_img = len(img_list)


tag_fn = join(data_root, dataset, dataset + '_i2v_tag.pkl')
with open(tag_fn, 'rb') as fp:
    tags = pickle.load(fp)
# num_img = tags.shape[0]

label = []
feat = []
cnt = 0

for i in range(num_img):

    # tag_bit1 = np.packbits(tags[i, :13])
    # tag_bit2 = np.packbits(tags[i, 13:23])
    # tag_bit3 = np.packbits(tags[i, 23:28])
    # tag_bit4 = np.packbits(tags[i, 28:])
    # tag_bit = np.concatenate((tag_bit1, tag_bit2, tag_bit3, tag_bit4), axis=0)

    # if os.path.exists(img_list[i]):
    #     img = cv2.imread(img_list[i])
    #     for j in range(num_img):
    #         if not i == j:
    #             if os.path.exists(img_list[j]):
    #                 target = cv2.imread(img_list[j])
    #                 diff = np.sum(img - target)
    #                 if diff == 0:
    #                     cnt += 1
    #                     shutil.move(img_list[j], join(save_path, img_list[j]))
    #                     print('%d duplicates' % cnt)

    # for j in range(n_tags):
    #     if tags[i][j] == 1:
    #         save_dir = join(save_path, tag_list[j])
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         shutil.copyfile(img_list[i], save_dir + '/' + os.path.basename(img_list[i]))

    img = Image.open(img_list[i])
    cnt += 1
    # img_path = join(data_path, '%05d.jpg' % (i+1))
    # # if os.path.exists(img_path):
    # #     cnt += 1
    # #     label.append(tags[i])
    # #     shutil.copyfile(img_path, save_path + '/%05d.jpg' % cnt)
    # #     print('Refining dataset...[%d/%d]' % (cnt, num_img))
    # gender = np.zeros(2, dtype=np.int64)
    # girl_img = join(save_path1, '%05d.jpg' % (i+1))
    # boy_img = join(save_path2, '%05d.jpg' % (i+1))
    #
    # if os.path.exists(girl_img):
    #     gender[0] = 1
    #     cnt += 1
    #     shutil.copyfile(img_path, save_path + '/%05d.jpg' % cnt)
    #     label.append(np.concatenate((gender, tags[i])))
    # elif os.path.exists(boy_img):
    #     gender[1] = 1
    #     cnt += 1
    #     shutil.copyfile(img_path, save_path + '/%05d.jpg' % cnt)
    #     label.append(np.concatenate((gender, tags[i])))
    # else:
    #     print("%05d.jpg is excluded" % (i+1))
    #     # pre_tags = illust2vec.estimate_specific_tags([img], [
    #     #     # "male",
    #     #     # "multiple boys",
    #     #     "1boy",
    #     #     # "multiple girls",
    #     #     "1girl"])[0]
    #     # pre_tags_list = list(pre_tags.values())
    #     # # if pre_tags_list[0] < 0.5 and pre_tags_list[1] < 0.5 and pre_tags_list[2] < 0.5:
    #     # #     if pre_tags_list[3] < pre_tags_list[4]:
    #     # if pre_tags_list[0] < pre_tags_list[1]:
    #     #     gender[0] = 1
    #     #     shutil.copyfile(img_path, save_path1 + '/%05d.jpg' % (i + 1))
    #     # else:
    #     #     gender[1] = 1
    #     #     cnt += 1
    #     #     shutil.copyfile(img_path, save_path2 + '/%05d.jpg' % (i + 1))
    #     #
    #     # shutil.copyfile(img_path, save_path + '/%05d.jpg' % (i+1))
    #
    # # print("Adding %d boy tags..(%d/%d)" % (cnt, i + 1, num_img))
    # print("Refining %d images..(%d/%d)" % (cnt, i + 1, num_img))




    # # detect and crop face region
    # cascade = cv2.CascadeClassifier(cascade_file)
    #
    # image = cv2.imread(img_list[i], cv2.IMREAD_COLOR)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    #
    # faces = cascade.detectMultiScale(gray)
    # img_h = image.shape[0]
    # img_w = image.shape[1]
    #
    # for (x, y, w, h) in faces:
    #     scale_factor = min((scale - 1.) / 2, x / w, y / h,
    #                        (img_w - x - w) / w, (img_h - y - h) / h)
    #     sw = int(w * scale_factor)
    #     sh = int(h * scale_factor)
    #
    #     xx = x + w + sw
    #     yy = y + h + sh
    #     x = x - sw
    #     y = y - sh
    #     assert (x >= 0 and y >= 0 and xx <= img_w and
    #             yy <= img_h), (w, h, image.shape, x, y, xx, yy)
    #
    #     if xx - x >= min_size and yy - y >= min_size:
    #         cnt += 1
    #         face = image[y:yy, x:xx, :]
    #         # img = cv2.resize(face, dsize=(resize_scale, resize_scale))
    #
    #         outname = os.path.join(save_path, '%05d.jpg' % cnt)
    #         cv2.imwrite(outname, face)
    #         print('Detecting faces...[%d/%d]' % (cnt, num_img))
    #
    #         img = Image.open(outname)
    #         pre_tags = illust2vec.estimate_specific_tags([img], [
    #                     "no humans",
    #                     "1girl",
    #                     "1boy"
    #             ])[0]
    #         pre_tags_list = list(pre_tags.values())
    #         if (pre_tags_list[1] < pre_tags_list[2]) or (pre_tags_list[0] > 0.25):
    #             shutil.copyfile(outname, save_path2 + '/%05d.jpg' % cnt)
    #         else:
    #             shutil.copyfile(outname, save_path1 + '/%05d.jpg' % cnt)
            # img = Image.open(outname)
    tags = illust2vec.estimate_specific_tags([img], [
            # "1girl",
            # "1boy",
            "blonde hair",
            "brown hair",
            "black hair",
            "blue hair",
            "pink hair",
            "purple hair",
            "green hair",
            "red hair",
            "silver hair",
            "white hair",
            "orange hair",
            "aqua hair",
            "grey hair",
            "blue eyes",
            "red eyes",
            "brown eyes",
            "green eyes",
            "purple eyes",
            "yellow eyes",
            "pink eyes",
            "aqua eyes",
            "black eyes",
            "orange eyes",
            "long hair",
            "short hair",
            "twintails",
            "drill hair",
            "ponytail",
            "dark skin",
            "blush",
            "smile",
            "open mouth",
            "hat",
            "ribbon",
            "glasses"
        ])[0]
    # max_val1 = np.max(list(tags.values())[:2])
    max_val2 = np.max(list(tags.values())[:13])
    max_val3 = np.max(list(tags.values())[13:23])

    idx = 0
    one_hot = []
    for val in list(tags.values()):
        # if idx < 2:
        #     if val == max_val1:
        #         one_hot.append(1)
        #     else:
        #         one_hot.append(0)
        if idx < 13:
            if val == max_val2:
                one_hot.append(1)
            else:
                one_hot.append(0)
        elif idx < 23:
            if val == max_val3:
                one_hot.append(1)
            else:
                one_hot.append(0)
        elif idx < 35:
            if val > 0.25:
                one_hot.append(1)
            else:
                one_hot.append(0)

        idx += 1
    label.append(one_hot)
    print('Estimating tags...[%d/%d]' % (cnt, num_img))

    #     feat.append(illust2vec2.extract_feature([img])[0])
    #     print('Extracting feats...[%d/%d]' % (cnt, num_img))
    # else:
    #     print('No face is detected!')
    #     else:
    #         print('Not 1girl!')
    # else:
    #     print('Not female!')

# label = tags

tag_fn = join(data_root, dataset, dataset + '_i2v_tag.pkl')
with open(tag_fn, 'wb') as fp:
    pickle.dump(np.array(label), fp)

# feat_fn = join(data_root, dataset, dataset + '_i2v_feat.pkl')
# with open(feat_fn, 'wb') as fp:
#     pickle.dump(np.array(feat), fp)

print('end')
#
# # In the feature vector extraction, you do not need to specify the tag.
# illust2vec = i2v.make_i2v_with_chainer("illust2vec_ver200.caffemodel")
#
# # illust2vec = i2v.make_i2v_with_caffe(
# #     "illust2vec.prototxt", "illust2vec_ver200.caffemodel")
#
# img = Image.open("images/miku.jpg")
#
# # extract a 4,096-dimensional feature vector
# result_real = illust2vec.extract_feature([img])
# print("shape: {}, dtype: {}".format(result_real.shape, result_real.dtype))
# print(result_real)
#
# # i2v also supports a 4,096-bit binary feature vector
# result_binary = illust2vec.extract_binary_feature([img])
# print("shape: {}, dtype: {}".format(result_binary.shape, result_binary.dtype))
# print(result_binary)