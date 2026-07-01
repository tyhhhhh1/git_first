import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantitativeAsymmetricLoss(nn.Module):
    """Pointwise shape loss + area compensation for quantitative spectra."""

    def __init__(self, fg_boost_ratio=3.0, bg_penalty_ratio=2.0, bg_threshold=1e-4, area_weight=1.0):
        super().__init__()
        self.fg_boost_ratio = fg_boost_ratio
        self.bg_penalty_ratio = bg_penalty_ratio
        self.bg_threshold = bg_threshold
        self.area_weight = area_weight

    def forward(self, pred, target):
        base_loss = F.l1_loss(pred, target, reduction='none')

        target_max = target.max(dim=-1, keepdim=True)[0]
        is_active_channel = (target_max > 1e-4).float()
        active_scale = 1.0 / torch.clamp(target_max, min=0.05)
        scale_factor = is_active_channel * active_scale + (1.0 - is_active_channel)

        is_background = (target <= self.bg_threshold).float()
        is_foreground = 1.0 - is_background
        weights = (is_foreground * self.fg_boost_ratio) + (is_background * self.bg_penalty_ratio)

        pointwise_loss = (base_loss * weights * scale_factor).mean()

        pred_area = F.relu(pred).sum(dim=-1)
        target_area = target.sum(dim=-1)
        area_loss = F.l1_loss(torch.log1p(pred_area), torch.log1p(target_area))

        return pointwise_loss + (self.area_weight * area_loss)


class MultiScaleReconstructionLoss(nn.Module):
    """Energy-conservation reconstruction loss at multiple smooth scales."""

    def __init__(self, scales=(1, 2, 4, 8)):
        super().__init__()
        self.scales = scales

    def forward(self, pred_sum, mixture):
        total_loss = 0.0
        for scale in self.scales:
            if scale == 1:
                total_loss = total_loss + F.l1_loss(pred_sum, mixture)
            else:
                p = F.avg_pool1d(pred_sum, kernel_size=scale, stride=scale)
                m = F.avg_pool1d(mixture, kernel_size=scale, stride=scale)
                total_loss = total_loss + F.l1_loss(p, m) / scale
        return total_loss


class PhysicsInformedLoss(nn.Module):
    """
    Fine-tuning loss for real/serum-like mixtures.

    This version emphasizes overlap resolution and weak-channel recall.
    """

    def __init__(self, pure_w=1.0, recon_w=1.0, presence_w=1.0, group_w=0.0,
                 fraction_w=0.0,
                 local_false_w=0.0, excess_w=0.0,
                 presence_threshold=0.02, target_presence_threshold=1e-4,
                 negative_presence_weight=1.55,
                 hard_channel_boost=2.0,
                 hard_channel_names=('Asparagine', 'Valine', 'Glutamine', 'Glutamate',
                                     'Isoleucine', 'Leucine', 'Proline'),
                 ratio_groups=(('Glutamine', 'Glutamate'),
                               ('Isoleucine', 'Leucine', 'Valine', 'Proline'),
                               ('Asparagine', 'Glutamine', 'Glutamate'),
                               ('Alanine', 'Serine', 'Glycine', 'Threonine')),
                 mapping=None):
        super().__init__()
        self.pure_w = pure_w
        self.recon_w = recon_w
        self.presence_w = presence_w
        self.group_w = group_w
        self.fraction_w = fraction_w
        self.local_false_w = local_false_w
        self.excess_w = excess_w
        self.presence_threshold = presence_threshold
        self.target_presence_threshold = target_presence_threshold
        self.negative_presence_weight = negative_presence_weight
        self.pure_loss_fn = QuantitativeAsymmetricLoss()
        self.recon_loss_fn = MultiScaleReconstructionLoss()
        self.hard_channel_boost = hard_channel_boost
        self.hard_indices = []
        self.ratio_group_indices = []
        if mapping is not None:
            self.hard_indices = [mapping[name] for name in hard_channel_names if name in mapping]
            for group in ratio_groups:
                idx = [mapping[name] for name in group if name in mapping]
                if len(idx) >= 2:
                    self.ratio_group_indices.append(idx)

    def _presence_loss(self, pred, target):
        pred_score = pred.amax(dim=-1)
        target_score = (target.amax(dim=-1) > self.target_presence_threshold).float()
        logits = (pred_score - self.presence_threshold) * 30.0

        weights = torch.ones_like(target_score)
        if self.hard_indices:
            weights[:, self.hard_indices] = self.hard_channel_boost

        class_weight = torch.where(
            target_score > 0,
            torch.full_like(target_score, 1.6),
            torch.full_like(target_score, self.negative_presence_weight),
        )
        return F.binary_cross_entropy_with_logits(logits, target_score, weight=weights * class_weight)

    def _group_ratio_loss(self, pred, target):
        if not self.ratio_group_indices:
            return pred.new_tensor(0.0)

        losses = []
        for indices in self.ratio_group_indices:
            pred_area = F.relu(pred[:, indices, :]).sum(dim=-1)
            target_area = target[:, indices, :].sum(dim=-1)
            target_total = target_area.sum(dim=-1, keepdim=True)
            active = (target_total.squeeze(-1) > 1e-4).float()
            if active.sum() <= 0:
                continue

            pred_total = torch.clamp(pred_area.sum(dim=-1, keepdim=True), min=1e-6)
            pred_ratio = pred_area / pred_total
            target_ratio = target_area / torch.clamp(target_total, min=1e-6)

            ratio_loss = torch.abs(pred_ratio - target_ratio).sum(dim=-1)
            area_loss = torch.abs(torch.log1p(pred_total.squeeze(-1)) - torch.log1p(target_total.squeeze(-1)))
            losses.append(((ratio_loss + 0.25 * area_loss) * active).sum() / (active.sum() + 1e-8))

        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _area_fraction_loss(self, pred, target):
        pred_area = F.relu(pred).sum(dim=-1)
        target_area = target.sum(dim=-1)
        active = (target_area > self.target_presence_threshold).float()
        if active.sum() <= 0:
            return pred.new_tensor(0.0)

        pred_total = torch.clamp(pred_area.sum(dim=-1, keepdim=True), min=1e-6)
        target_total = torch.clamp(target_area.sum(dim=-1, keepdim=True), min=1e-6)
        pred_fraction = pred_area / pred_total
        target_fraction = target_area / target_total

        weights = active
        if self.hard_indices:
            weights[:, self.hard_indices] = weights[:, self.hard_indices] * self.hard_channel_boost
        return (torch.abs(pred_fraction - target_fraction) * weights).sum() / (weights.sum() + 1e-8)

    def _local_false_peak_loss(self, pred, target, support_window=31):
        """Penalize peaks inside active channels but outside target-supported ppm regions."""
        pred_pos = F.relu(pred)
        target_support = (target > self.target_presence_threshold).float()
        pooled_support = F.max_pool1d(
            target_support.reshape(-1, 1, target.shape[-1]),
            kernel_size=support_window,
            stride=1,
            padding=support_window // 2,
        ).reshape_as(target)
        false_region = 1.0 - pooled_support
        active = (target.amax(dim=-1, keepdim=True) > self.target_presence_threshold).float()
        weighted_false_region = false_region * active
        return (pred_pos * weighted_false_region).sum() / (weighted_false_region.sum() + 1e-8)

    def _mixture_excess_loss(self, pred, background, mixture):
        """Penalize summed predictions that exceed the measured mixture pointwise."""
        pred_sum = F.relu(pred).sum(dim=1, keepdim=True) + F.relu(background)
        available = torch.clamp(mixture, min=0.0)
        return F.relu(pred_sum - available).mean()

    def forward(self, pred, target, background, mixture):
        l_pure = self.pure_loss_fn(pred, target)

        pred_sum = pred.sum(dim=1, keepdim=True) + background
        l_recon = self.recon_loss_fn(pred_sum, mixture)
        l_presence = self._presence_loss(pred, target)
        l_group = self._group_ratio_loss(pred, target)
        l_fraction = self._area_fraction_loss(pred, target)
        l_local_false = self._local_false_peak_loss(pred, target)
        l_excess = self._mixture_excess_loss(pred, background, mixture)

        total_loss = (
            self.pure_w * l_pure
            + self.recon_w * l_recon
            + self.presence_w * l_presence
            + self.group_w * l_group
            + self.fraction_w * l_fraction
            + self.local_false_w * l_local_false
            + self.excess_w * l_excess
        )
        loss_dict = {
            'total': total_loss.item(),
            'l_pure': l_pure.item(),
            'l_recon': l_recon.item(),
            'l_presence': l_presence.item(),
            'l_group': l_group.item(),
            'l_fraction': l_fraction.item(),
            'l_local_false': l_local_false.item(),
            'l_excess': l_excess.item(),
        }
        return total_loss, loss_dict
