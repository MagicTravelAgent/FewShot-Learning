import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.transforms.functional import to_pil_image
import pandas as pd
import argparse

from docs.HSNet.Model.HSNet import HypercorrSqueezeNetwork
from docs.HSNet.Common import Utils
from docs.HSNet.Common.Visualizer import Visualizer
from docs.HSNet.Common.Evaluator import Evaluator
from docs.HSNet.DataLoader.FSSDataset import FSSDataset

from docs.MSANet.test import get_model
from docs.MSANet.test import get_parser

def test_MSANet_loop(model, dataloader):
    r""" Test MSANet """

    # Freeze randomness during testing for reproducibility
    Utils.fix_randseed(0)

    ious = []
    eval = []
    for idx, batch in tqdm(enumerate(dataloader)):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = Utils.to_cuda(batch)
        pred_mask, _, _ = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], cat_idx=None)
        pred_mask = pred_mask.max(1)[1]

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union, eval_dict = Evaluator.classify_prediction(pred_mask.clone(), batch)
        iou = area_inter[1].float() / area_union[1].float()
        eval.append(eval_dict)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, idx, iou_b=iou)
        ious.append(iou[0].float().item())
        print(iou[0].float().item())
    print("Mean IOU", np.array(ious).mean())
    return eval

def MSANet_test(args_args, args, dataloader_test):

    args_msa = get_parser(args_args)
    # Model initialization
    model = get_model(args_msa)
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model) # It says that this doesn't exist for the version of tf I use, but when I remove it the whole thing breaks so ¯\_(ツ)_/¯
    model.to(device)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args["visualize"])

    # Test MSANet
    with torch.no_grad():
        eval_list = test_MSANet_loop(model, dataloader_test)
    
    # eval_list saved for analysis
    df = pd.DataFrame.from_dict(eval_list)
    df.to_csv("docs/output/MSA_top20.csv")

def test_HSNet_loop(model, dataloader, nshot, confidence):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    Utils.fix_randseed(0)

    eval = []
    for idx, batch in tqdm(enumerate(dataloader)):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = Utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union, eval_dict = Evaluator.classify_prediction(pred_mask.clone(), batch)
        iou = area_inter[1].float() / area_union[1].float()
        eval.append(eval_dict)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, idx, iou_b=iou)
    print("Segmentation Complete")
    return eval

def HSNet_test(args, dataloader_test):

    # Model initialization
    model = HypercorrSqueezeNetwork(
        args["backbone"], args["use_original_imgsize"])
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model) # It says that this doesn't exist for the version of tf I use, but when I remove it the whole thing breaks so ¯\_(ツ)_/¯
    model.to(device)

    # Load trained model
    if args["load"] == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args["load"]))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args["visualize"])

    # Test HSNet
    with torch.no_grad():
        eval_list = test_HSNet_loop(model, dataloader_test, args["nshot"], args["confidence_level"])
    
    # eval_list saved for analysis
    df = pd.DataFrame.from_dict(eval_list)
    df.to_csv("docs/output/HSNet_80_full.csv")


def test_CNN_loop(model, dataloader, confidence):
    r""" Test HSNet """

    ious = []
    eval = []
    for idx, batch in tqdm(enumerate(dataloader)):

        # 1. Networks forward pass
        output = model(batch['query_img'])["out"].softmax(dim=1)

        pred_mask = torch.unsqueeze(output[0, 15], 0)
        pred_mask[pred_mask < confidence] = 0
        pred_mask[pred_mask >= confidence] = 1

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union, eval_dict = Evaluator.classify_prediction(pred_mask.clone(), batch)
        iou = area_inter[1].float() / area_union[1].float()
        eval.append(eval_dict)

        pred_mask = to_pil_image(pred_mask)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_CNN(idx, batch['query_img'][0], batch['query_mask'][0], pred_mask, iou=iou)
        ious.append(iou[0].float().item())
    print("Segmentation Complete")
    return eval



def CNN_test(args, dataloader_test):

    # Load the pretrained segmentation model
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    # Set the model to eval mode
    model.eval()

     # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args["visualize"])

    # Test CNN
    with torch.no_grad():
        eval_list = test_CNN_loop(model, dataloader_test, args["confidence_level"])
    
    # eval_list saved for analysis
    df = pd.DataFrame.from_dict(eval_list)
    df.to_csv("docs/output/CNN.csv")


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Few Shot Semantic Segmentation')
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'HSNet', 'MSANet'])
    parser.add_argument('--datapath', type=str, default='docs/Data/')
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--benchmark', type=str, default='custom')
    parser.add_argument('--load', type=str, default='docs/HSNet/Model/res101_pas/res101_pas_fold3/best_model.pt')
    parser.add_argument('--nshot', type=int, default=5)
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--use_original_imgsize', type=bool, default=False)
    parser.add_argument('--confidence_level', type=int, default=0.5)

    # ============== MSANet Parsing just ignore these ==============
    parser.add_argument('--arch', type=str, default='MSANet')
    parser.add_argument('--viz', action='store_true', default=True)
    # parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='docs\\MSANet\\config\\pascal\\pascal_split2_resnet101.yaml',
                        help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()

    arg_dict = vars(args)

    # Dataset initialization
    FSSDataset.initialize(
        img_size=400, datapath=arg_dict["datapath"], use_original_imgsize=arg_dict["use_original_imgsize"], length=arg_dict["test_size"])
    dataloader_test = FSSDataset.build_dataloader(
        benchmark=arg_dict["benchmark"], experiment='test', shot=arg_dict["nshot"])

    if(arg_dict["model"] == "CNN"):
        CNN_test(arg_dict, dataloader_test)

    if(arg_dict["model"] == "HSNet"):
        HSNet_test(arg_dict, dataloader_test)

    if(arg_dict["model"] == "MSANet"):
        MSANet_test(args, arg_dict, dataloader_test)