import argparse
import json
import os, shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader as original
from utils.dataloaders_clahe import create_dataloader as clahe
from utils.dataloaders_agcwd import create_dataloader as agcwd
from utils.dataloaders_aod import create_dataloader as aod
from utils.dataloaders_clahe_aod import create_dataloader as clahe_aod
from utils.dataloaders_agcwd_aod import create_dataloader as agcwd_aod
from utils.dataloaders_aod_clahe import create_dataloader as aod_clahe
from utils.dataloaders_aod_agcwd import create_dataloader as aod_agcwd
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou, fitness
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
import json
import pandas as pd

try:
    os.mkdir('output')
except:
    shutil.rmtree('output')
    os.mkdir('output')

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        input_json,
        data='D:\\dataset\\yolov5\\bcc.yaml', 
        batch_size=2,  # batch size
        imgsz=704,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=0,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None
):
    input_file = open(input_json)
    user_input = json.load(input_file)
    models = {}
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        weights = []
        for enhancement in user_input:
            for weight in user_input[enhancement]:
                if weight not in weights:
                    weights.append(weight)
        
        for weight in weights:
            models[weight] = DetectMultiBackend(weight, device=device, dnn=dnn, data=data, fp16=half)
        print(len(models))
        first = next(iter(models))
        stride, pt, jit, engine = models[first].stride, models[first].pt, models[first].jit, models[first].engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = models[first].fp16  # FP16 supported on limited backends with CUDA
        
        if engine:
            batch_size = models[first].batch_size
        else:
            device = models[first].device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    for model in models:
        models[model].eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Dataloader
    dataloaders = {}
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = models[first].model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        for model in models:
            models[model].warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        for enhancement in user_input:
            dataloaders[enhancement] = globals()[enhancement](data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = models[first].names if hasattr(models[first], 'names') else models[first].module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    stats_d = {}
    for i in user_input:
        pbar = tqdm(dataloaders[i], desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        stats_a = []
        for model in user_input[i]:
            stats = []
            callbacks.run('on_val_start')
            for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
                callbacks.run('on_val_batch_start')
                with dt[0]:
                    if cuda:
                        im = im.to(device, non_blocking=True)
                        targets = targets.to(device)
                    im = im.half() if half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    nb, _, height, width = im.shape  # batch size, channels, height, width

                # Inference
                with dt[1]:
                    preds, train_out = models[model](im) if compute_loss else (models[model](im, augment=augment), None)

                # Loss
                if compute_loss:
                    loss += compute_loss(train_out, targets)[1]  # box, obj, cls

                # NMS
                targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                with dt[2]:
                    preds = non_max_suppression(preds,
                                                conf_thres,
                                                iou_thres,
                                                labels=lb,
                                                multi_label=True,
                                                agnostic=single_cls,
                                                max_det=max_det)

                # Metrics
                for si, pred in enumerate(preds):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    path, shape = Path(paths[si]), shapes[si][0]
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                    seen += 1

                    if npr == 0:
                        if nl:
                            stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                            if plots:
                                confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                        continue

                    # Predictions
                    if single_cls:
                        pred[:, 5] = 0
                    predn = pred.clone()
                    scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                    # Evaluate
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = process_batch(predn, labelsn, iouv)
                        if plots:
                            confusion_matrix.process_batch(predn, labelsn)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)
            #print(stats)
            stats_a.append(stats)
        stats_d[i] = stats_a
        
    # Compute metrics
    results = {}
    for stats_a in stats_d:
        results_a = []
        for stats in stats_d[stats_a]:
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))
            results_a.append([mp, mr, map50, map, fi[0]])
        results[stats_a] = results_a

    df = [["Enhancement method", "Weight", "Precision", "Recall", "mAP50", "mAP50-95", "fitness"]]
    best_algo = ""
    best_fit = 0
    for i in user_input:
        cnt = 0
        for j in user_input[i]:
            df.append([i, j, results[i][cnt][0],results[i][cnt][1], results[i][cnt][2], results[i][cnt][3], results[i][cnt][4]])
            if(results[i][cnt][4] > best_fit):
                best_fit = results[i][cnt][4]
                best_algo = i+" enhancement + "+j+" weight file"
            cnt += 1
    df = pd.DataFrame(df)
    df.columns = df.iloc[0]
    df.drop(df.index[0],inplace=True)
    print("Browse output/output.csv for detailed data")
    df.to_csv('output/output.csv', index=False)
    print("Recommended combination is " + best_algo)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='D:\\dataset\\yolov5\\my_input.json', help='input.json path')
    parser.add_argument('--data', type=str, default='D:\\dataset\\yolov5\\bcc.yaml', help='dataset.yaml path')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=704, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    #print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        return run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

