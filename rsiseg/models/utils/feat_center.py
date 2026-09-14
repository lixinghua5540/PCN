import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchsnooper
import time

#device = torch.device("cuda" if not args.cpu else "cpu")
device = torch.device("cuda")

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())



# def class_center_cal(feature, labels, num_classes, eps=1e-6):
#     """Calculate class centers with improved numerical stability
    
#     Args:
#         feature: input features [N, C, H, W]
#         labels: ground truth labels [N, 1, H, W]
#         num_classes: number of classes
#         eps: small value for numerical stability
#     """
#     # Process labels
#     labels = labels.to(device)
#     labels = torch.squeeze(labels, 1).long()  # [N, H, W]
#     n, h, w = labels.shape
    
#     # Downsample labels to match feature size
#     feature_size = feature.shape[2:]
#     if (h, w) != feature_size:
#         labels = F.interpolate(labels.unsqueeze(1).float(), 
#                               size=feature_size, 
#                               mode='nearest').squeeze(1).long()
    
#     # Flatten spatial dimensions
#     labels = labels.view(n, -1)  # [N, H*W]
#     feature = feature.view(n, feature.shape[1], -1)  # [N, C, H*W]
    
#     # Handle ignore label (255)
#     valid_mask = (labels != 255)
#     labels = labels * valid_mask.long()  # set invalid to 0
    
#     # Calculate class counts
#     count_class = torch.zeros(num_classes, device=device)
#     for i in range(num_classes):
#         count_class[i] = (labels == i).sum()
    
#     # Calculate class centers
#     class_center = torch.zeros(num_classes, feature.shape[1], device=device)
#     for i in range(num_classes):
#         if count_class[i] > 0:
#             # Select features belonging to class i
#             class_mask = (labels == i).float()
#             class_features = torch.matmul(class_mask.unsqueeze(1), feature.transpose(1, 2))
#             class_center[i] = class_features.squeeze(0) / (count_class[i] + eps)
    
#     return class_center  # [num_classes, C]

def class_center_cal(feature, labels, num_classes, eps=1e-6):
    """Calculate class centers with improved accuracy and stability
    
    Args:
        feature: input features [N, C, H, W] (raw features, no softmax)
        labels: ground truth labels [N, 1, H, W]
        num_classes: number of classes (excluding ignore label)
        eps: small value for numerical stability
        
    Returns:
        Tensor of shape [num_classes, C] containing class centers
    """
    # Validate inputs
    assert feature.dim() == 4, "Features must be 4D tensor"
    assert labels.dim() == 4, "Labels must be 4D tensor"
    
    # Process labels
    labels = labels.to(device).squeeze(1).long()  # [N, H, W]
    n, h, w = labels.shape
    n, c, fh, fw = feature.shape
    
    # Align feature and label sizes
    if (h, w) != (fh, fw):
        labels = F.interpolate(labels.unsqueeze(1).float(),
                             size=(fh, fw),
                             mode='nearest').squeeze(1).long()
    
    # Flatten spatial dimensions
    labels_flat = labels.view(n, -1)  # [N, fh*fw]
    features_flat = feature.view(n, c, -1)  # [N, C, fh*fw]
    
    # Calculate class centers efficiently
    class_center = torch.zeros(num_classes, c, device=device)
    counts = torch.zeros(num_classes, device=device)
    
    for cls in range(num_classes):
        # Create mask for current class
        mask = (labels_flat == cls).float()  # [N, fh*fw]#is this mask corresponding to the features exactly
        valid = (mask.sum(dim=1) > 0)  # for each batch of data, samples containing this class
        
        if valid.any():#not all the samples are false
            # Weighted sum of features for this class
            sum_features = torch.einsum('ncv,nv->c', 
                                      features_flat[valid], 
                                      mask[valid])
            sum_counts = mask[valid].sum()
            
            class_center[cls] = sum_features / (sum_counts + eps)
            counts[cls] = sum_counts
    
    return class_center

def class_center_cal1(features, labels, num_classes, eps=1e-6):
    """Calculate class centers with improved accuracy and stability
    
    Args:
        feature: input features [N, C, H, W] (raw features, no softmax)
        labels: ground truth labels [N, 1, H, W]
        num_classes: number of classes (excluding ignore label)
        eps: small value for numerical stability
        
    Returns:
        Tensor of shape [num_classes, C] containing class centers
    """
    # Validate inputs
    # assert feature.dim() == 4, "Features must be 4D tensor"
    # assert labels.dim() == 4, "Labels must be 4D tensor"
    class_centers = []
    for feature in features:
    # Process labels
        labels = labels.to(device).squeeze(1).long()  # [N, H, W]
        n, h, w = labels.shape
        n, c, fh, fw = feature.shape
    
        # Align feature and label sizes
        if (h, w) != (fh, fw):
            labels = F.interpolate(labels.unsqueeze(1).float(),
                             size=(fh, fw),
                             mode='nearest').squeeze(1).long()
    
        # Flatten spatial dimensions
        labels_flat = labels.view(n, -1)  # [N, fh*fw]
        features_flat = feature.view(n, c, -1)  # [N, C, fh*fw]
    
        # Calculate class centers efficiently
        class_center = torch.zeros(num_classes, c, device=device)
        counts = torch.zeros(num_classes, device=device)
    
        for cls in range(num_classes):
        # Create mask for current class
            mask = (labels_flat == cls).float()  # [N, fh*fw]#is this mask corresponding to the features exactly
            valid = (mask.sum(dim=1) > 0)  # for each batch of data, samples containing this class
        
            if valid.any():#not all the samples are false
                # Weighted sum of features for this class
                sum_features = torch.einsum('ncv,nv->c', 
                                      features_flat[valid], 
                                      mask[valid])
                sum_counts = mask[valid].sum()
            
                class_center[cls] = sum_features / (sum_counts + eps)
                counts[cls] = sum_counts
        class_centers.append(class_center)
    return class_centers
def class_center_cal1_improved(features, labels, num_classes, eps=1e-6):
    """Calculate class centers with improved efficiency"""
    # 预处理标签（只需一次）
    labels = labels.to(device).squeeze(1).long()
    
    class_centers = []
    for feature in features:
        n, c, fh, fw = feature.shape
        n_labels, h, w = labels.shape
        
        # 对齐尺寸
        if (h, w) != (fh, fw):
            resized_labels = F.interpolate(
                labels.unsqueeze(1).float(), 
                size=(fh, fw), 
                mode='nearest'
            ).squeeze(1).long()
        else:
            resized_labels = labels
            
        # 扁平化
        labels_flat = resized_labels.view(n, -1)  # [N, fh*fw]
        features_flat = feature.view(n, c, -1)    # [N, C, fh*fw]
        
        # 一次性计算所有类中心
        class_center = torch.zeros(num_classes, c, device=device)
        counts = torch.zeros(num_classes, device=device)
        
        for cls in range(num_classes):
            mask = (labels_flat == cls)  # 布尔掩码，更节省内存
            valid_mask = mask.any(dim=1)  # 包含该类别的样本
            
            if valid_mask.any():
                # 使用布尔索引而不是浮点掩码
                masked_features = features_flat[valid_mask]  # [K, C, L]
                masked_labels = labels_flat[valid_mask]      # [K, L]
                
                # 为当前类别创建精确掩码
                cls_mask = (masked_labels == cls).float()
                
                # 计算加权和
                sum_features = torch.einsum('ncl,nl->c', masked_features, cls_mask)
                sum_counts = cls_mask.sum()
                
                class_center[cls] = sum_features / (sum_counts + eps)
                counts[cls] = sum_counts
                
        class_centers.append(class_center)
    
    return class_centers

def class_center_cal_with_external_selection(features, labels, selection_mask, num_classes, eps=1e-6):
    """Calculate class centers using externally computed selection mask
    
    Args:
        features: list of input features [N, C, H, W] (raw features, no softmax)
        labels: pseudo labels [N, 1, H, W]
        selection_mask: binary mask [N, 1, H, W] indicating which pixels to use (1=use, 0=ignore)
        num_classes: number of classes (excluding ignore label)
        eps: small value for numerical stability
        
    Returns:
        List of tensors of shape [num_classes, C] containing class centers
    """
    labels = labels.to(device).squeeze(1).long()  # [N, H, W]
    selection_mask = selection_mask.to(device).squeeze(1)  # [N, H, W]
    
    class_centers = []
    
    for feature in features:
        n, c, fh, fw = feature.shape
        n_labels, h, w = labels.shape
        
        if (h, w) != (fh, fw):
            resized_labels = F.interpolate(
                labels.unsqueeze(1).float(), 
                size=(fh, fw), 
                mode='nearest'
            ).squeeze(1).long()
            
            resized_selection = F.interpolate(
                selection_mask.unsqueeze(1).float(), 
                size=(fh, fw), 
                mode='nearest'
            ).squeeze(1)
        else:
            resized_labels = labels
            resized_selection = selection_mask
            
        class_center = torch.zeros(num_classes, c, device=device)
        counts = torch.zeros(num_classes, device=device)
        
        for cls in range(num_classes):
            cls_mask = (resized_labels == cls)
            combined_mask = cls_mask & (resized_selection > 0.5) 
            
            valid_samples = combined_mask.any(dim=1).any(dim=1)  # [N]
            
            if valid_samples.any():
                for i in range(n):
                    if not valid_samples[i]:
                        continue

                    sample_cls_mask = cls_mask[i]  # [H, W]
                    sample_combined_mask = combined_mask[i]  # [H, W]
                    sample_features = feature[i]  # [C, H, W]
                    
                    selected_features = sample_features[:, sample_combined_mask]  # [C, K]
                    
                    if selected_features.size(1) > 0:
                        sum_features = selected_features.sum(dim=1)
                        sum_count = selected_features.size(1)
                        class_center[cls] += sum_features
                        counts[cls] += sum_count

            if counts[cls] > eps:
                class_center[cls] /= counts[cls]
        
        class_centers.append(class_center)
    
    return class_centers


def class_center_cal_with_external_selection_org(features, labels, selection_mask, num_classes, eps=1e-6):
    """Calculate class centers using externally computed selection mask
    
    Args:
        features: list of input features [N, C, H, W] (raw features, no softmax)
        labels: pseudo labels [N, 1, H, W]
        selection_mask: binary mask [N, 1, H, W] indicating which pixels to use (1=use, 0=ignore)
        num_classes: number of classes (excluding ignore label)
        eps: small value for numerical stability
        
    Returns:
        List of tensors of shape [num_classes, C] containing class centers
    """
# Validate inputs
    # assert feature.dim() == 4, "Features must be 4D tensor"
    # assert labels.dim() == 4, "Labels must be 4D tensor"
    class_centers = []
    for feature in features:
    # Process labels
        labels = labels.to(device).squeeze(1).long()  # [N, H, W]
        selection_mask = selection_mask.to(device).squeeze(1)  # [N, H, W]
        n, h, w = labels.shape
        n, c, fh, fw = feature.shape
    
        # Align feature and label sizes
        if (h, w) != (fh, fw):
            labels = F.interpolate(labels.unsqueeze(1).float(),
                             size=(fh, fw),
                             mode='nearest').squeeze(1).long()
            
            selection_mask = F.interpolate(selection_mask.unsqueeze(1).float(),
                                     size=(fh, fw),
                                     mode='nearest').squeeze(1)
        # Flatten spatial dimensions
        labels_flat = labels.view(n, -1)  # [N, fh*fw]
        features_flat = feature.view(n, c, -1)  # [N, C, fh*fw]
        selection_flat = selection_mask.view(n, -1)  # [N, fh*fw]
        # Calculate class centers efficiently
        class_center = torch.zeros(num_classes, c, device=device)
        counts = torch.zeros(num_classes, device=device)
    
        for cls in range(num_classes):
        # Create mask for current class
            mask = (labels_flat == cls).float() * (selection_flat > 0.5).float() # [N, fh*fw]#is this mask corresponding to the features exactly
            valid = (mask.sum(dim=1) > 0)  # for each batch of data, samples containing this class
        
            if valid.any():#not all the samples are false
                # Weighted sum of features for this class
                sum_features = torch.einsum('ncv,nv->c', 
                                      features_flat[valid], 
                                      mask[valid])
                sum_counts = mask[valid].sum()
            
                class_center[cls] = sum_features / (sum_counts + eps)
                counts[cls] = sum_counts
        class_centers.append(class_center)
    return class_centers


def class_center_update(class_center_source, class_center_source_ori, p, num_classes):
    """Update class centers with momentum
    
    Args:
        class_center_source: Current class centers [num_classes, c]
        class_center_source_ori: Original class centers [num_classes, c] 
        p: Momentum parameter
        num_classes: Number of classes
    """
    for i in range(num_classes):
        valid_features = class_center_source[i][class_center_source[i] > 0]#为什么选大于零的部分
        if len(valid_features) > 0:
            # Update with momentum
            class_center_source[i] = p * class_center_source[i] +  (1 - p) * class_center_source_ori[i].float()#.detach()
        else:
            # Keep original if no valid features
            class_center_source[i] = class_center_source_ori[i]#.detach()
    return class_center_source


# def class_center_precal(feature,labels,num_classes=19):
#     count_class = np.zeros((19, 1))
#     labels = labels.to(device)
#     n,c,h,w = feature.size()
#     labels = F.interpolate(labels.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).long()
#     n_l,h_l,w_l = labels.size()
#     labels = labels.view(n_l,h_l*w_l)
#     labels_for_count = labels.cpu().data[0].numpy()
#     labels_for_count[labels_for_count==255] = 19  #255->19
#     count_class= np.bincount(labels_for_count, minlength=19)
#     count_class = count_class[:19]
#     count_class = torch.from_numpy(count_class).unsqueeze(1)                                      
#     labels[labels==255] = 19  #255->19
#     labels = torch.unsqueeze(labels,2).cpu()
#     labels = torch.zeros(n,h_l*w_l,num_classes+1).scatter_(2,labels,1)
#     labels = labels.transpose(1,2) #n 20 h*w
#     labels = labels[:,:num_classes,:]  #n 19 h*w
#     feature = nn.functional.interpolate(feature,size=(h_l, w_l), mode='bilinear', align_corners=True)
#     feature = F.softmax(feature,1)                    #softmax
#     feature = feature.view(n,c,-1)
#     feature = feature.transpose(1,2).cpu()  #n h*w c
#     class_center = torch.matmul(labels,feature)  #n 19 c

#     return class_center,count_class

def class_center_precal(feature, labels, num_classes):
    """Precompute class centers and counts
    
    Args:
        feature: Input features [n, c, h, w]
        labels: Ground truth labels [n, h, w]
        num_classes: Number of classes
    """
    labels = labels.to(device)
    n, c, h, w = feature.size()
    
    # Downsample labels
    labels = F.interpolate(labels.unsqueeze(1).float(), 
                          scale_factor=0.5, 
                          mode='nearest').squeeze(1).long()
    # labels = F.interpolate(labels.unsqueeze(1).float(), 
    #                       scale_factor=1, 
    #                       mode='nearest').squeeze(1).long()
    # Process labels
    n_l, h_l, w_l = labels.size()
    labels = labels.view(n_l, h_l * w_l)
    
    # Count class occurrences (handle ignore label 255)
    labels_for_count = labels.cpu().data[0].numpy()
    labels_for_count[labels_for_count == 255] = num_classes
    count_class = np.bincount(labels_for_count, minlength=num_classes)#？？？ minlength=num_classes + 1
    count_class = count_class[:num_classes]  # Exclude ignore class
    count_class = torch.from_numpy(count_class).unsqueeze(1)
    
    # Create one-hot encoding
    labels[labels == 255] = num_classes  # Map ignore label to num_classes
    labels = torch.unsqueeze(labels, 2).cpu()
    labels = torch.zeros(n, h_l * w_l, num_classes + 1).scatter_(2, labels, 1)
    labels = labels.transpose(1, 2)  # n x (num_classes+1) x h*w
    labels = labels[:, :num_classes, :]  # Exclude ignore class
    
    # Process features
    feature = nn.functional.interpolate(feature, 
                                      size=(h_l, w_l), 
                                      mode='bilinear', 
                                      align_corners=True)
    feature = F.softmax(feature, 1)  # softmax
    feature = feature.view(n, c, -1)
    feature = feature.transpose(1, 2).cpu()  # n x h*w x c
    
    # Compute class centers
    class_center = torch.matmul(labels, feature)  # n x num_classes x c
    
    return class_center, count_class