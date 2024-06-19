import json
import numpy as np
import math
import torch

import scipy.stats as stats
from skimage.io import imsave
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage import morphology
from scipy import ndimage


def map_eval(pred, gt, map_type, mask,  aggregate_output_dict):

    assert pred.shape == gt.shape

    rmse = np.sqrt(np.mean((gt - pred)**2))
    mae = np.mean(np.abs(gt - pred))

    gt_flat = gt[mask==1]
    pred_flat = pred[mask==1]

    scc, scc_pval = stats.spearmanr(pred_flat, gt_flat, nan_policy='omit')
    pcc, pcc_pval = stats.pearsonr(pred_flat, gt_flat)

    # print('scc shape', scc.shape)
    # print('pcc shape', pcc.shape)

    # print('scc_pval', scc_pval)
    # print('pcc_pval', pcc_pval)

    ssim_by_slice = []
    psnr_by_slice = []
    # lpips_by_slice = []
    
    height = pred.shape[-1]

    if map_type == 'tmax':
        data_range = 24
    elif map_type == 'cbv':
        data_range = 200
    elif map_type == 'cbf':
        data_range = 1000
    else:
        raise NotImplementedError

    for h in range(height):
        # lpips_by_slice.append(calculate_lpips(pred[:,:,h], gt[:,:,h]))
        ssim_by_slice.append(SSIM(gt[:,:,h], pred[:,:,h], gaussian_weights = True, sigma = 1.5, use_sample_covariance = False, data_range = data_range))
        psnr_by_slice.append(PSNR(gt[:,:,h], pred[:,:,h], data_range = data_range))

    ssim = np.mean(ssim_by_slice)
    psnr = np.mean(psnr_by_slice)
    # lpips = np.mean(lpips_by_slice)
    # ssim = ssim / height
    # psnr = psnr / height
    
    aggregate_output_dict[map_type]['rmse'].append(rmse)
    aggregate_output_dict[map_type]['mae'].append(mae)
    aggregate_output_dict[map_type]['scc'].append(scc)
    aggregate_output_dict[map_type]['pcc'].append(pcc)
    aggregate_output_dict[map_type]['ssim'].append(ssim)
    aggregate_output_dict[map_type]['psnr'].append(psnr)
    # aggregate_output_dict[map_type]['lpips'].append(lpips)


    metric_values = {
        'rmse':rmse,
        'mae':mae,
        'scc':(scc, scc_pval),
        'pcc':(pcc, pcc_pval),
        'ssim_by_slice':ssim_by_slice,
        'psnr_by_slice':psnr_by_slice,
        # 'lpips_by_slice':lpips_by_slice,
        'ssim':ssim,
        'psnr':psnr,
        # 'lpips':lpips
    }

    return metric_values

def save_01_img(save_result_path, arr):
    # print()
    # print(save_result_path, np.min(arr), np.max(arr))

    assert np.min(arr) >= 0
    assert np.max(arr) <= 1

    imsave(save_result_path, (arr*255).astype(np.uint8))

def tensor2np2D(tensor):
    return tensor.cpu().data.numpy().squeeze()

def test_single_case(model, sampled_batch, stride_xy,  stride_z, patch_size):

    gt_tmax = np.array(sampled_batch['tmax'].squeeze())
    gt_cbv = np.array(sampled_batch['cbv'].squeeze())
    gt_cbf = np.array(sampled_batch['cbf'].squeeze())

    mask = (gt_tmax + gt_cbv + gt_cbf) > 0
    mask = morphology.remove_small_objects(mask.astype(bool), 1000)
    mask = ndimage.morphology.binary_fill_holes(mask)



    data = sampled_batch['data']
    name = sampled_batch['name'][0]


    b, dd, ww, hh = data.shape
    p0,p1,p2 = patch_size

    sx = math.ceil((ww - p0) / stride_xy) + 1
    sy = math.ceil((hh - p1) / stride_xy) + 1
    sz = math.ceil((dd - p2) / stride_z) + 1

    empty_arr = np.zeros([ww,hh]).astype(np.float32)
    cnt = np.zeros([ww,hh]).astype(np.float32)
    pred_collection = {k:empty_arr.copy() for k in ['pred_tmax', 'pred_cbv', 'pred_cbf']}
    for x in range(0,sx):
        xs = min(stride_xy*x, ww-p0)
        for y in range(0,sy):
            ys = min(stride_xy * y, hh-p1)
            for z in range(0,sz):
                zs = min(stride_z * z, dd-p2)

                # patch = data[:, zs:zs+p2, xs:xs+p0, ys:ys+p1].cuda()
                patch = data[:, zs:zs+p2, xs:xs+p0, ys:ys+p1].cuda()

                with torch.no_grad():
                    out_dict, _, _ = model(patch)

                pred_collection['pred_tmax'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_dict['tmax'])
                pred_collection['pred_cbv'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_dict['cbv'])
                pred_collection['pred_cbf'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_dict['cbf'])
                cnt[xs:xs+p0, ys:ys+p1] += 1
    
    for k,v in pred_collection.items():
        pred_collection[k] = v/cnt


    test_result = {
        'name':name,
        'pred_tmax':pred_collection['pred_tmax'] * mask,
        'gt_tmax':gt_tmax,
        'pred_cbv':pred_collection['pred_cbv'] * mask,
        'gt_cbv':gt_cbv,
        'pred_cbf':pred_collection['pred_cbf'] * mask,
        'gt_cbf':gt_cbf,
        'mask':mask
    }

    return test_result


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)