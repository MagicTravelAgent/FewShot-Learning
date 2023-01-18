import torch

class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask')
        query_name = batch.get("query_name")
        print(query_name)
        output_dict = {}
        # here collect information about the query mask: size, globules (and their size), 
        output_dict["query_name"] = query_name[0]
        output_dict["y_mask_size"] = gt_mask.sum().item()

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            
            output_dict["pred_mask_size"] = pred_mask.sum().item()

            _inter = _pred_mask[_pred_mask == _gt_mask]

            output_dict["intersection_size"] = _inter.sum().item()
            output_dict["wrong_pred_size"] =  output_dict["pred_mask_size"] - output_dict["intersection_size"]

            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()


        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        output_dict["union"] = area_union[1].float()
        output_dict["IOU"] = output_dict["intersection_size"] / output_dict["union"]
        output_dict["percent_y"] = output_dict["intersection_size"] / output_dict["pred_mask_size"]

        # return a dictionary containing the whole evaluation

        print(output_dict)
        return area_inter, area_union, output_dict