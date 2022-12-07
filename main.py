import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from docs.HSNet.Model.HSNet import HypercorrSqueezeNetwork
from docs.HSNet.Common import Utils
from docs.HSNet.Common.Visualizer import Visualizer
from docs.HSNet.Common.Evaluator import Evaluator
from docs.HSNet.DataLoader.FSSDataset import FSSDataset


def test(model, dataloader, nshot):
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
        "bsz": 1,
        "lr": 1e-3,
        "nworker": 0,
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
    with torch.no_grad():
        test_miou = test(model, dataloader_test, args["nshot"])
    print("Mean IOU:", test_miou)

HSNet_test()