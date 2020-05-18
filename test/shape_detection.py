import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torchvision
from torch.utils.data import DataLoader

from karakara import config
config.GPU = False
import karakara.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout, BatchNormalization_v2
from karakara.layers import Input, Add, Separate, Same, Flatten, Activation, Transpose, Reshape
from karakara.layers import Conv2D, MaxPooling2D, AveragePooling2DAll
from karakara.activations import Sigmoid, ReLU, LeakyReLU, Softmax, MaskedSigmoid
from karakara.optimizers import SGD, Momentum, Adam
from karakara.regulizers import l2
from karakara.callbacks import ReduceLROnPlateau

from karakara.losses import BaseLossLayer, MeanSquareError, BinaryCrossEntropy, CategoricalCrossEntropy

input_shape = (3, 112, 112)
n_classes = 2
anchors = np.array([(0.2, 0.2), (2.8, 1.4), (1.4, 2.8)])
epochs = 200
batch_size = 8


def add_conv2d_bn_leaky(model, num_filters=16, kernel_size=3, strides=1):
    model.add(Conv2D(num_filters, kernel_size, strides, padding='same', use_bias=False))
    model.add(BatchNormalization_v2())
    model.add(LeakyReLU(0.1))


def make_darknet19(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    add_conv2d_bn_leaky(model, 32, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 64, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 128, 3)
    add_conv2d_bn_leaky(model, 64, 1)
    add_conv2d_bn_leaky(model, 128, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 256, 3)
    add_conv2d_bn_leaky(model, 128, 1)
    add_conv2d_bn_leaky(model, 256, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 512, 3)
    add_conv2d_bn_leaky(model, 256, 1)
    add_conv2d_bn_leaky(model, 512, 3)
    add_conv2d_bn_leaky(model, 256, 1)
    add_conv2d_bn_leaky(model, 512, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 1024, 3)
    add_conv2d_bn_leaky(model, 512, 1)
    add_conv2d_bn_leaky(model, 1024, 3)
    add_conv2d_bn_leaky(model, 512, 1)
    add_conv2d_bn_leaky(model, 1024, 3)

    add_conv2d_bn_leaky(model, 1000, 1)

    model.add(Conv2D(5 * (4 + 1 + 1), 1, 1))

    return model


def make_my_yolo(input_shape, num_anchors, num_classes):
    ori_c, ori_w, ori_h = input_shape

    model = Sequential()
    model.add(Input(shape=input_shape))

    add_conv2d_bn_leaky(model, 16, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 32, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 64, 3)
    model.add(MaxPooling2D(2, 2))

    add_conv2d_bn_leaky(model, 128, 3)
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(num_anchors * (4 + 1 + num_classes), 1, 1))
    model.add(Reshape((num_anchors, (4 + 1 + num_classes), ori_w // 16, ori_h // 16)))
    model.add(Transpose((0, 1, 3, 4, 2)))

    mask = np.ones((3, 7, 7, 7), dtype=bool)
    mask[..., 2] = False
    mask[..., 3] = False
    model.add(MaskedSigmoid(mask))

    return model


class yolo_loss(BaseLossLayer):
    def __init__(self, scaled_anchors, ignore_th=0.5, obj_scale=1, noobj_scale=100):
        super().__init__()
        self.x_mse = MeanSquareError()
        self.y_mse = MeanSquareError()
        self.w_mse = MeanSquareError()
        self.h_mse = MeanSquareError()
        self.obj_conf_bce = BinaryCrossEntropy()
        self.noobj_conf_bce = BinaryCrossEntropy()
        self.cls_bce = BinaryCrossEntropy()

        self.anchors = scaled_anchors
        self.num_anchors = len(scaled_anchors)
        self.ignore_th = ignore_th
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale

    def get_iou_WH(self, wh1, wh2):
        wh2 = wh2.T
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = np.minimum(w1, w2) * np.minimum(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        return inter_area / union_area

    def get_yolo_targets(self, pred_info, target, anchors, ignore_th):
        batch_size, num_anchors, grid_size, num_cls = pred_info

        sizeT = batch_size, num_anchors, grid_size, grid_size
        obj_mask = np.zeros(sizeT, dtype='bool')
        noobj_mask = np.ones(sizeT, dtype='bool')
        tx = np.zeros(sizeT)
        ty = np.zeros(sizeT)
        tw = np.zeros(sizeT)
        th = np.zeros(sizeT)

        sizeT = batch_size, num_anchors, grid_size, grid_size, num_cls
        tcls = np.zeros(sizeT)

        target_bboxes = target[:, 2:] * grid_size
        t_x = target_bboxes[:, 0]
        t_y = target_bboxes[:, 1]
        t_wh = target_bboxes[:, 2:]
        t_w = target_bboxes[:, 2]
        t_h = target_bboxes[:, 3]

        grid_i = target_bboxes[:, 0].astype('int')
        grid_j = target_bboxes[:, 1].astype('int')

        iou_with_anchors = [self.get_iou_WH(anchor, t_wh) for anchor in self.anchors]
        iou_with_anchors = np.stack(iou_with_anchors)
        best_anchor_ind = iou_with_anchors.argmax(0)

        batch_inds, target_labels = target[:, :2].astype('int').T
        target_labels -= 1
        obj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 1
        noobj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 0

        for ind, iou_wa in enumerate(iou_with_anchors.T):
            noobj_mask[batch_inds[ind], iou_wa > ignore_th, grid_j[ind], grid_i[ind]] = 0

        tx[batch_inds, best_anchor_ind, grid_j, grid_i] = t_x - np.floor(t_x)
        ty[batch_inds, best_anchor_ind, grid_j, grid_i] = t_y - np.floor(t_y)

        anchor_w = anchors[best_anchor_ind][:, 0]
        tw[batch_inds, best_anchor_ind, grid_j, grid_i] = np.log(t_w / anchor_w + 1e-16)

        anchor_h = anchors[best_anchor_ind][:, 1]
        th[batch_inds, best_anchor_ind, grid_j, grid_i] = np.log(t_h / anchor_h + 1e-16)

        tcls[batch_inds, best_anchor_ind, grid_j, grid_i, target_labels] = 1

        output = {
            "obj_mask": obj_mask,
            "noobj_mask": noobj_mask,
            "tx": tx,
            "ty": ty,
            "tw": tw,
            "th": th,
            "tcls": tcls,
            "t_conf": obj_mask.astype('float32'),
        }
        return output

    def call(self, yolo_out, labels, **kwargs):
        self.pred = yolo_out

        batch_size, _, grid_size, _, _ = yolo_out.shape

        x = yolo_out[:, :, :, :, 0]
        y = yolo_out[:, :, :, :, 1]
        w = yolo_out[:, :, :, :, 2]
        h = yolo_out[:, :, :, :, 3]
        pred_conf = yolo_out[:, :, :, :, 4]
        pred_cls_prob = yolo_out[:, :, :, :, 5:]

        pred_info = (batch_size, self.num_anchors, grid_size, pred_cls_prob.shape[-1])
        yolo_targets = self.get_yolo_targets(pred_info, labels, self.anchors, self.ignore_th)

        obj_mask = yolo_targets["obj_mask"]
        noobj_mask = yolo_targets["noobj_mask"]
        self.obj_mask = obj_mask
        self.noobj_mask = noobj_mask
        tx = yolo_targets["tx"]
        ty = yolo_targets["ty"]
        tw = yolo_targets["tw"]
        th = yolo_targets["th"]
        tcls = yolo_targets["tcls"]
        t_conf = yolo_targets["t_conf"]

        # print()
        # print(pred_conf[noobj_mask])
        # print(t_conf[noobj_mask])

        loss_x = self.x_mse.call(x[obj_mask], tx[obj_mask])
        loss_y = self.y_mse.call(y[obj_mask], ty[obj_mask])
        loss_w = self.w_mse.call(w[obj_mask], tw[obj_mask])
        loss_h = self.h_mse.call(h[obj_mask], th[obj_mask])

        loss_conf_obj = self.obj_conf_bce.call(pred_conf[obj_mask], t_conf[obj_mask])
        loss_conf_noobj = self.noobj_conf_bce.call(pred_conf[noobj_mask], t_conf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.cls_bce.call(pred_cls_prob[obj_mask], tcls[obj_mask])
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        print()
        print(f'loss_x: {loss_x}, loss_y: {loss_y}, loss_w: {loss_w}, loss_h: {loss_h}')
        print(f'loss_obj: {self.obj_scale * loss_conf_obj}, loss_noobj: {self.noobj_scale * loss_conf_noobj}, loss_conf: {loss_conf}, loss_cls: {loss_cls}')

        return loss

    def backward(self, dout=1):
        dyolo_out = np.zeros_like(self.pred)

        dx = dyolo_out[:, :, :, :, 0]
        dy = dyolo_out[:, :, :, :, 1]
        dw = dyolo_out[:, :, :, :, 2]
        dh = dyolo_out[:, :, :, :, 3]
        dpred_conf = dyolo_out[:, :, :, :, 4]
        dpred_cls_prob = dyolo_out[:, :, :, :, 5:]

        dx[self.obj_mask] = self.x_mse.backward()[0]
        dy[self.obj_mask] = self.y_mse.backward()[0]
        dw[self.obj_mask] = self.w_mse.backward()[0]
        dh[self.obj_mask] = self.h_mse.backward()[0]
        dpred_conf[self.obj_mask] = self.obj_scale * self.obj_conf_bce.backward()[0]
        dpred_conf[self.noobj_mask] = self.noobj_scale * self.noobj_conf_bce.backward()[0]
        dpred_cls_prob[self.obj_mask] = self.cls_bce.backward()[0]

        return -dyolo_out


def main():

    class bbox_transform:
        def __init__(self, img_size):
            self.img_size = img_size

        def __call__(self, x):
            bbox_num = len(x)
            target = np.zeros((bbox_num, 6))

            for index, bbox in enumerate(x):
                target[index, 1] = bbox['category_id']
                target[index, 2:] = bbox['bbox']

            target[:, 2:] /= self.img_size

            return target

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((112, 112)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    training_data = torchvision.datasets.CocoDetection(
        '../datasets/shape/train', '../datasets/shape/annotations/instances_train.json', transform=transform_train, target_transform=bbox_transform(512))
    testing_data = torchvision.datasets.CocoDetection(
        '../datasets/shape/val', '../datasets/shape/annotations/instances_val.json', transform=transform_train, target_transform=bbox_transform(512))

    def collate_fn(batch):
        imgs, targets = list(zip(*batch))

        # Remove empty boxes
        targets = [boxes for boxes in targets if boxes is not None]

        # set the sample index
        for b_i, boxes in enumerate(targets):
            boxes[:, 0] = b_i
        targets = np.concatenate(targets, 0)
        imgs = np.stack([img for img in imgs])
        return imgs, targets

    training_laoder = DataLoader(training_data, batch_size=8, collate_fn=collate_fn)

    loss_params = {
        "scaled_anchors": anchors,
        "ignore_th": 0.5,
        "obj_scale": 1,
        "noobj_scale": 100,
    }

    model = make_my_yolo(input_shape, len(anchors), n_classes)
    model.summary()
    model.compile(Adam(lr=0.001), yolo_loss(**loss_params), None)

    history = model.fit_dataloader(training_laoder, batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    main()
