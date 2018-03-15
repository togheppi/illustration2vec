import cv2
import numpy as np
import os
from anime_face_detector import detect


"""checking kernel size"""
def odd(x):
    if x % 2 == 0:
        return x + 1
    else:
        return x


crop_size = 64
resize_scale = 128

kernel_size = 3

num_iter = 1

# Directories for loading data and saving results
# dataset = 'steam'
# dataset = 'safebooru'
dataset = 'getchu'
data_dir = '../Data/' + dataset + '/'

img_list = sorted(os.listdir(data_dir))

cnt = 0
for i in range(len(img_list)):
    img_fn = img_list[i]
    if dataset == 'steam':
        if img_fn > '2':
            img = cv2.imread(data_dir + img_fn)

            # # center crop
            # c_x = (img.shape[0] - crop_size) // 2
            # c_y = (img.shape[1] - crop_size) // 2
            # face = img[c_x:c_x + crop_size, c_y:c_y + crop_size]

            # save_dir = '../Data/' + dataset + '_color_cs%d/train/' % crop_size
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # cv2.imwrite(save_dir + img_fn, face)

            ##### original edge detector
            src_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            src_img_gray_float = src_img_gray / 255.

            ks_blur = (int)(np.min(np.shape(src_img_gray)) // 80)

            ks_edge = ks_blur // 2

            """ Edge Detection """
            kernel = np.ones((ks_edge, ks_edge), dtype=np.uint8)
            dilation = cv2.dilate(src_img_gray, kernel, iterations=1)
            inv_edge = cv2.subtract(dilation, src_img_gray)
            src_img_edge = 255 - inv_edge
            src_img_edge_float = src_img_edge / 255.

            save_dir1 = '../Data/' + dataset + '_edge_cs%d_ks%d_it%d/train/' % (crop_size, ks_edge, num_iter)
            if not os.path.exists(save_dir1):
                os.mkdir(save_dir1)

            """ Sketch Enhancement """
            # get invert image
            src_img_gray_inv = 255 - src_img_gray

            # conduct gaussian blur
            src_img_gray_inv_blur = cv2.GaussianBlur(src_img_gray_inv, ksize=(odd(ks_blur), odd(ks_blur)), sigmaX=0,
                                                     sigmaY=0)
            count = 1
            while True:
                ks = odd(ks_blur // (count))

                if ks >= 3:
                    src_img_gray_inv_blur = np.minimum(src_img_gray_inv_blur,
                                                       cv2.GaussianBlur(src_img_gray_inv, ksize=(ks, ks),
                                                                        sigmaX=0, sigmaY=0))
                    count += 1
                else:
                    break

            src_img_gray_inv_blur_float = src_img_gray_inv_blur / 255.

            src_img_gray_blend_float = src_img_edge_float
            for _ in range(1):
                src_img_edge_float = src_img_gray_blend_float

                # get blended image
                src_img_gray_blend_float = np.where(
                    np.logical_and(src_img_edge_float < 0.99, src_img_gray_inv_blur_float < 0.85),
                    (src_img_edge_float) * src_img_gray_float / (
                        1 - src_img_gray_inv_blur_float), src_img_edge_float)

                # enhance contrast
                src_img_gray_blend_float = np.clip(src_img_gray_blend_float, 0.0, 1)
                x = src_img_gray_blend_float
                v = 0.2
                s1 = 0.1
                s2 = (1 - s1 * v) / (1 - v)
                x = np.where(x < v, s1 * x, s2 * (x - v) + s1 * v)

                src_img_gray_blend_float = x

                src_img_gray_blend_float = np.clip(src_img_gray_blend_float, 0.0, 1)

            rst_img = ((src_img_gray_blend_float) * 255).astype(np.uint8)

            save_dir2 = '../Data/' + dataset + '_edge_enhanced_cs%d_ks%d_it%d/train/' % (crop_size, ks_edge, num_iter)
            if not os.path.exists(save_dir2):
                os.mkdir(save_dir2)

            # center crop
            c_x = (img.shape[0] - crop_size) // 2
            c_y = (img.shape[1] - crop_size) // 2
            edge_face1 = src_img_edge[c_x:c_x + crop_size, c_y:c_y + crop_size]
            edge_face2 = rst_img[c_x:c_x + crop_size, c_y:c_y + crop_size]
            # resize
            edge_face1 = cv2.resize(edge_face1, (resize_scale, resize_scale), interpolation=cv2.INTER_LINEAR)
            edge_face2 = cv2.resize(edge_face2, (resize_scale, resize_scale), interpolation=cv2.INTER_LINEAR)

            # save results
            # cv2.imwrite(save_dir1 + img_fn, edge_face1)
            cv2.imwrite(save_dir2 + img_fn, edge_face2)

            if cnt % 1000 == 0:
                print('%d image is processed.' % cnt)

            cnt += 1
    elif dataset == 'safebooru':
        img = cv2.imread(data_dir + img_fn)

        ##### original edge detector
        src_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        src_img_gray_float = src_img_gray / 255.

        ks_blur = (int)(np.min(np.shape(src_img_gray)) // 80)

        ks_edge = ks_blur // 2

        if ks_edge == 1:
            ks_edge += 1

        """ Edge Detection """
        kernel = np.ones((ks_edge, ks_edge), dtype=np.uint8)
        dilation = cv2.dilate(src_img_gray, kernel, iterations=1)
        inv_edge = cv2.subtract(dilation, src_img_gray)
        src_img_edge = 255 - inv_edge
        src_img_edge_float = src_img_edge / 255.

        """ Sketch Enhancement """
        # get invert image
        src_img_gray_inv = 255 - src_img_gray

        # conduct gaussian blur
        src_img_gray_inv_blur = cv2.GaussianBlur(src_img_gray_inv, ksize=(odd(ks_blur), odd(ks_blur)), sigmaX=0,
                                                 sigmaY=0)
        count = 1
        while True:
            ks = odd(ks_blur // (count))

            if ks >= 3:
                src_img_gray_inv_blur = np.minimum(src_img_gray_inv_blur,
                                                   cv2.GaussianBlur(src_img_gray_inv, ksize=(ks, ks),
                                                                    sigmaX=0, sigmaY=0))
                count += 1
            else:
                break

        src_img_gray_inv_blur_float = src_img_gray_inv_blur / 255.

        src_img_gray_blend_float = src_img_edge_float
        for _ in range(1):
            src_img_edge_float = src_img_gray_blend_float

            # get blended image
            src_img_gray_blend_float = np.where(
                np.logical_and(src_img_edge_float < 0.99, src_img_gray_inv_blur_float < 0.85),
                (src_img_edge_float) * src_img_gray_float / (
                    1 - src_img_gray_inv_blur_float), src_img_edge_float)

            # enhance contrast
            src_img_gray_blend_float = np.clip(src_img_gray_blend_float, 0.0, 1)
            x = src_img_gray_blend_float
            v = 0.2
            s1 = 0.1
            s2 = (1 - s1 * v) / (1 - v)
            x = np.where(x < v, s1 * x, s2 * (x - v) + s1 * v)

            src_img_gray_blend_float = x

            src_img_gray_blend_float = np.clip(src_img_gray_blend_float, 0.0, 1)

        rst_img = ((src_img_gray_blend_float) * 255).astype(np.uint8)

        # detect and crop face region
        face_img, (x, y, w, h) = detect(data_dir + img_fn)
        if face_img is not None:
            # face color
            face = img[y:y + h, x:x + w]
            # resize
            face = cv2.resize(face, (resize_scale, resize_scale), interpolation=cv2.INTER_CUBIC)

            save_dir1 = '../Data/' + dataset + '_color/train/'
            if not os.path.exists(save_dir1):
                os.mkdir(save_dir1)

            cv2.imwrite(save_dir1 + img_fn, face)

            # face edge
            edge = rst_img[y:y + h, x:x + w]
            # resize
            edge = cv2.resize(edge, (resize_scale, resize_scale), interpolation=cv2.INTER_CUBIC)

            save_dir2 = '../Data/' + dataset + '_edge_enhanced/train/'
            if not os.path.exists(save_dir2):
                os.mkdir(save_dir2)

            cv2.imwrite(save_dir2 + img_fn, edge)

            print('%d image is processed.' % cnt)
            cnt += 1

    elif dataset == 'getchu':
        img = cv2.imread(data_dir + img_fn)

        # ##### original edge detector
        # src_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # src_img_gray_float = src_img_gray / 255.
        #
        # ks_blur = (int)(np.min(np.shape(src_img_gray)) // 80)
        #
        # ks_edge = ks_blur // 2
        #
        # if ks_edge == 1:
        #     ks_edge += 1
        #
        # """ Edge Detection """
        # kernel = np.ones((ks_edge, ks_edge), dtype=np.uint8)
        # dilation = cv2.dilate(src_img_gray, kernel, iterations=1)
        # inv_edge = cv2.subtract(dilation, src_img_gray)
        # src_img_edge = 255 - inv_edge
        # src_img_edge_float = src_img_edge / 255.
        #
        # """ Sketch Enhancement """
        # # get invert image
        # src_img_gray_inv = 255 - src_img_gray
        #
        # # conduct gaussian blur
        # src_img_gray_inv_blur = cv2.GaussianBlur(src_img_gray_inv, ksize=(odd(ks_blur), odd(ks_blur)), sigmaX=0,
        #                                          sigmaY=0)
        # count = 1
        # while True:
        #     ks = odd(ks_blur // (count))
        #
        #     if ks >= 3:
        #         src_img_gray_inv_blur = np.minimum(src_img_gray_inv_blur,
        #                                            cv2.GaussianBlur(src_img_gray_inv, ksize=(ks, ks),
        #                                                             sigmaX=0, sigmaY=0))
        #         count += 1
        #     else:
        #         break
        #
        # src_img_gray_inv_blur_float = src_img_gray_inv_blur / 255.
        # src_img_gray_blend_float = src_img_edge_float
        #
        # for _ in range(1):
        #     src_img_edge_float = src_img_gray_blend_float
        #
        #     # get blended image
        #     src_img_gray_blend_float = np.where(
        #         np.logical_and(src_img_edge_float < 0.99, src_img_gray_inv_blur_float < 0.85),
        #         (src_img_edge_float) * src_img_gray_float / (
        #             1 - src_img_gray_inv_blur_float), src_img_edge_float)
        #
        #     # enhance contrast
        #     src_img_gray_blend_float = np.clip(src_img_gray_blend_float, 0.0, 1)
        #     x = src_img_gray_blend_float
        #     v = 0.2
        #     s1 = 0.1
        #     s2 = (1 - s1 * v) / (1 - v)
        #     x = np.where(x < v, s1 * x, s2 * (x - v) + s1 * v)
        #
        #     src_img_gray_blend_float = x
        #
        #     src_img_gray_blend_float = np.clip(src_img_gray_blend_float, 0.0, 1)
        #
        # rst_img = ((src_img_gray_blend_float) * 255).astype(np.uint8)

        # detect and crop face region
        ratio = 1.5
        face_img, (x, y, w, h) = detect(data_dir + img_fn)
        if face_img is not None:
            # face color

            scale_w = int(w * ratio)
            scale_h = int(h * ratio)
            margin_w = (scale_w - w)//2
            margin_h = (scale_h - h)//2
            new_x = max(x - margin_w, 0)
            new_y = max(y - margin_h, 0)
            face = img[new_y:new_y + scale_h, new_x:new_x + scale_w]
            # resize
            face = cv2.resize(face, (resize_scale, resize_scale), interpolation=cv2.INTER_CUBIC)

            save_dir1 = '../Data/' + dataset + '_MoeGAN/train/'
            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)

            cv2.imwrite(save_dir1 + img_fn, face)

            # # face edge
            # edge = rst_img[y:y + h, x:x + w]
            # # resize
            # edge = cv2.resize(edge, (resize_scale, resize_scale), interpolation=cv2.INTER_CUBIC)
            #
            # save_dir2 = '../Data/' + dataset + '_edge_enhanced/train/'
            # if not os.path.exists(save_dir2):
            #     os.makedirs(save_dir2)
            #
            # cv2.imwrite(save_dir2 + img_fn, edge)

            print('%d image is processed.' % cnt)
            cnt += 1




