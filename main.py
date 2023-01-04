import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.transforms.functional import to_pil_image

from docs.HSNet.Model.HSNet import HypercorrSqueezeNetwork
from docs.HSNet.Common import Utils
from docs.HSNet.Common.Visualizer import Visualizer
from docs.HSNet.Common.Evaluator import Evaluator
from docs.HSNet.DataLoader.FSSDataset import FSSDataset


def test_HSNet_loop(model, dataloader, nshot):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    Utils.fix_randseed(0)

    ious = []
    for idx, batch in tqdm(enumerate(dataloader)):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = Utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask.clone(), batch)
        # print("IOU:",area_inter/area_union)
        iou = area_inter[1].float() / area_union[1].float()

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, idx, iou_b=iou)
        ious.append(iou[0].float().item())
    return np.array(ious).mean()

def HSNet_test():
    # ===============================================================================
    # Let's a-go!
    # ===============================================================================
    # Arguments parsing
    args = {
        "datapath": 'docs/Data/',
        "benchmark": 'custom',  # dataloader selection
        "load": "docs/HSNet/Model/res101_pas/res101_pas_fold3/best_model.pt",
        "nshot": 7,
        "backbone": 'resnet101',  # choices=['vgg16', 'resnet50', 'resnet101']
        "visualize": True,
        "use_original_imgsize": True
    }

    # Model initialization
    model = HypercorrSqueezeNetwork(
        args["backbone"], args["use_original_imgsize"])
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args["load"] == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args["load"]))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args["visualize"])

    # Dataset initialization
    FSSDataset.initialize(
        img_size=400, datapath=args["datapath"], use_original_imgsize=args["use_original_imgsize"])
    dataloader_test = FSSDataset.build_dataloader(
        benchmark=args["benchmark"], experiment='test', shot=args["nshot"])

    # Test HSNet
    test_miou = []
    with torch.no_grad():
        test_miou = test_HSNet_loop(model, dataloader_test, args["nshot"])
    print("Mean IOU:", test_miou)



def test_CNN_loop(model, dataloader, confidence):
    r""" Test HSNet """

    ious = []
    for idx, batch in tqdm(enumerate(dataloader)):

        # 1. Networks forward pass
        output = model(batch['query_img'])["out"].softmax(dim=1)

        pred_mask = torch.unsqueeze(output[0, 15], 0)
        pred_mask[pred_mask < confidence] = 0
        pred_mask[pred_mask >= confidence] = 1

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        iou = area_inter[1].float() / area_union[1].float()

        pred_mask = to_pil_image(pred_mask)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_CNN(idx, batch['query_img'][0], batch['query_mask'][0], pred_mask, iou=iou)
        ious.append(iou[0].float().item())
    return np.array(ious).mean()



def CNN_test():
    args = {
        "datapath": 'docs/Data/',
        "benchmark": 'custom',  # dataloader selection
        "visualize": True,
        "use_original_imgsize": False,
        "confidence level": 0.5
    }

    # Load the pretrained segmentation model
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    # Set the model to eval mode
    model.eval()

     # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args["visualize"])

    # Dataset initialization
    FSSDataset.initialize(
        img_size=400, datapath=args["datapath"], use_original_imgsize=args["use_original_imgsize"])
    dataloader_test = FSSDataset.build_dataloader(
        benchmark=args["benchmark"], experiment='test', shot=1)

    # Test CNN
    test_miou = []
    with torch.no_grad():
        test_miou = test_CNN_loop(model, dataloader_test, args["confidence level"])
    print("Mean IOU:", test_miou) 



HSNet_test()
# CNN_test()