
import segmentation_models_pytorch as smp
import torch

class F1ScorePatch(smp.utils.base.Metric):
    def __init__(self, threshold=0.5, activation=None, patch_thr=0.25, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = smp.base.modules.Activation(activation)
        self.patch_thr = patch_thr
        self.patch_size = patch_size

    def forward(self, y_pr, y_gt):  # pr => predicted, gt => groundtruth
        y_pr = self.activation(y_pr)
        y_pr = (y_pr > self.threshold).float()  # value 0.0 or 1.0

        y_gt = self.activation(y_gt)
        y_gt = (y_gt > self.threshold).float()  # value 0.0 or 1.0

        batch_size, nb_channels, height, width = y_pr.size()

        y_pr_patch_tensor = torch.zeros(batch_size, nb_channels, height//self.patch_size, width//self.patch_size)  # N, C, H, W
        y_gt_patch_tensor = torch.zeros(batch_size, nb_channels, height//self.patch_size, width//self.patch_size)

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                # Extract the patches of the batch
                patches_pr = y_pr[..., y:y + self.patch_size, x:x + self.patch_size]
                patches_gt = y_gt[..., y:y + self.patch_size, x:x + self.patch_size]

                # Iterate through each patch of the prediction
                for i, patch_pr in enumerate(patches_pr):
                    # Calculate the average of the patch
                    patch_avg_pr = torch.mean(patch_pr)
                    # Patch threshold
                    if patch_avg_pr > self.patch_thr:
                        y_pr_patch_tensor[i][0][y//self.patch_size][x//self.patch_size] = 1
                    else:
                        y_pr_patch_tensor[i][0][y//self.patch_size][x//self.patch_size] = 0

                # Iterate through each patch of the groundtruth mask
                for i, patch_gt in enumerate(patches_gt):
                    # Calculate the average of the patch
                    patch_avg_gt = torch.mean(patch_gt)
                    # Patch threshold
                    if patch_avg_gt > self.patch_thr:
                        y_gt_patch_tensor[i][0][y//self.patch_size][x//self.patch_size] = 1
                    else:
                        y_gt_patch_tensor[i][0][y//self.patch_size][x//self.patch_size] = 0

        tp, fp, fn, tn = smp.metrics.get_stats(y_pr_patch_tensor.to(torch.int), y_gt_patch_tensor.to(torch.int), mode='binary', threshold=self.threshold)
        f1_score_patch = smp.metrics.f1_score(tp=tp, fp=fp, fn=fn, tn=tn, reduction="micro") #reduction="micro" or "micro-imagewise"

        return f1_score_patch