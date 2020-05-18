import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))

import numpy as np
np.seterr(all='raise')
np.random.seed(2000)

import torch
torch.manual_seed(2000)
import torchvision
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw

from karakara import config
config.GPU = True
import karakara.backend as K
K.set_random_seed(22)
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from karakara.models import load_model
from karakara.optimizers import SGD, Momentum, Adam

from yolo_utils import make_my_yolo, bbox_transform, yolo_loss, get_yolo_targets

input_shape = (3, 224, 224)
n_classes = 2
anchors = np.array([[0.19, 0.17]])
anchors *= 7
epochs = 20
batch_size = 32


def draw_bboxes(img_data, bboxes, name, index, draw_grid=True):
    img_data = np.transpose(img_data * 255, (1, 2, 0)).astype('uint8')
    pil_img = Image.fromarray(img_data)
    draw = ImageDraw.Draw(pil_img)

    if draw_grid:
        step_count = 7
        y_start = 0
        y_end = pil_img.height
        step_size = int(pil_img.width / step_count)

        for x in range(0, pil_img.width + 1, step_size):
            if x == pil_img.width:
                x -= 1
            line = ((x, y_start), (x, y_end))
            draw.line(line, fill=(200,) * 3)

        x_start = 0
        x_end = pil_img.width

        for y in range(0, pil_img.height + 1, step_size):
            if y == pil_img.height:
                y -= 1
            line = ((x_start, y), (x_end, y))
            draw.line(line, fill=(200,) * 3)

    for cx, cy, w, h, color in bboxes:
        draw.rectangle(((cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2)),
                       outline=color, width=1)
        draw.rectangle(((cx - 1, cy - 1), (cx + 1, cy + 1)),
                       outline=(0, 255, 0), width=1)

    pil_img = pil_img.resize((500, 500))
    filename = f'result/{name}{index}.png'
    pil_img.save(filename, dpi=(500, 500))
    print(f'Save image to {filename}')


def eval(model, test_data, threshold=0.5, name='test'):
    preds = model.predict(test_data)

    for index, (pred, img_data) in enumerate(zip(preds, test_data)):
        bboxes = []
        for n_anchor, grid_h, grid_w in np.ndindex(pred.shape[:-1]):
            if pred[n_anchor, grid_h, grid_w, 4] > threshold:
                x, y, w, h, conf, c1, c2 = pred[n_anchor, grid_h, grid_w]
                cx = (grid_w + x) * 32
                cy = (grid_h + y) * 32
                w = (np.exp(w) * anchors[n_anchor][0]) * 32
                h = (np.exp(h) * anchors[n_anchor][1]) * 32
                color = (0, 0, 255) if c1 > c2 else (255, 0, 0)
                print(round(cx, 2), round(cy, 2), round(w, 2), round(h, 2), round(conf, 2))
                bboxes.append((cx, cy, w, h, color))

        draw_bboxes(img_data, bboxes, name, index)


def main():

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

    training_laoder = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    loss_params = {
        "scaled_anchors": anchors,
        "ignore_th": 0.4,
        "obj_scale": 1,
        "noobj_scale": 100,
    }

    def lr_schduler(model):
        lr = 1e-4
        if model.n_epoch > 26:
            lr *= 3e-3
        elif model.n_epoch > 18:
            lr *= 1e-2
        elif model.n_epoch > 12:
            lr *= 3e-2
        elif model.n_epoch > 3:
            lr *= 1e-1
        model.optimizer.lr = lr
        print('Learning rate: ', lr)
        return lr

    model = load_model('yolo.h5')
    # model = make_my_yolo(input_shape, len(anchors), n_classes)
    # model.summary()
    # model.compile(Momentum(lr=1e-4), yolo_loss(**loss_params), None)
    # history = model.fit_dataloader(training_laoder, batch_size=batch_size, epochs=epochs, callbacks=[lr_schduler])

    # target_laoder = DataLoader(training_data, batch_size=1, collate_fn=collate_fn, shuffle=False)
    # for index, (target_data, target_label) in enumerate(target_laoder):
    #     target_data = target_data
    #     target_label = target_label
    #     target = get_yolo_targets((1, len(anchors), 7, 2), target_label, anchors, 0.5)

    #     mask = target['obj_mask']
    #     bboxes = []
    #     for mask_index in np.argwhere(mask):
    #         mask_index = tuple(mask_index)
    #         _, n_anchor, grid_h, grid_w = mask_index
    #         print(n_anchor)
    #         x, y = target['tx'][mask_index], target['ty'][mask_index]
    #         w, h = target['tw'][mask_index], target['th'][mask_index]

    #         cx = (grid_w + x) * 32
    #         cy = (grid_h + y) * 32
    #         w = (np.exp(w) * anchors[n_anchor][0]) * 32
    #         h = (np.exp(h) * anchors[n_anchor][1]) * 32

    #         bboxes.append((cx, cy, w, h, (0, 0, 255)))

    #     draw_bboxes(target_data[0], bboxes, name='target', index=index)

    #     if index > 10:
    #         exit(0)

    test_data = np.array([testing_data[i][0].numpy() for i in range(20)])
    eval(model, test_data, name='test')

    test_data = np.array([training_data[i][0].numpy() for i in range(20)])
    eval(model, test_data, name='train')

    model.save('yolo.h5')


if __name__ == '__main__':
    main()
