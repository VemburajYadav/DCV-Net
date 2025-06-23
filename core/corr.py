import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, coords_grid
import altcorr


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, cosine_sim=False,
                 cv_softmax=False, lookup_softmax=False, lookup_softmax_all=False,
                 global_matching=False):
        self.num_levels = num_levels
        self.radius = radius
        self.cosine_sim = cosine_sim
        self.corr_pyramid = []
        self.lookup_softmax = lookup_softmax
        self.lookup_softmax_all = lookup_softmax_all

        # all pairs correlation
        corr = self.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        if global_matching:
            self.corrMap = corr.view(batch, h1 * w1, h2 * w2)

        if cv_softmax:
            corr = F.softmax(corr.view(batch * h1 * w1, dim, -1), dim=-1)
            corr = corr.view(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            _, dim, h2, w2 = corr.shape
            if cv_softmax:
                corr = F.softmax(corr.view(batch * h1 * w1, dim, -1), dim=-1)
                corr = corr.view(batch * h1 * w1, dim, h2, w2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)

            if self.lookup_softmax:
                corr = F.softmax(corr, dim=-1)

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)

        if self.lookup_softmax_all:
            out = F.softmax(out, dim=-1)
            
        return out.permute(0, 3, 1, 2).contiguous().float()

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 

        if self.cosine_sim:
            fmap1_mag = torch.linalg.vector_norm(fmap1, dim=1, keepdim=True) + 1e-8
            fmap2_mag = torch.linalg.vector_norm(fmap2, dim=1, keepdim=True) + 1e-8
            fmap1 = fmap1 / fmap1_mag
            fmap2 = fmap2 / fmap2_mag
            corr = torch.matmul(fmap1.transpose(1,2), fmap2)
            corr = corr.view(batch, ht, wd, 1, ht, wd)
            return corr
        else:
            corr = torch.matmul(fmap1.transpose(1,2), fmap2)
            corr = corr.view(batch, ht, wd, 1, ht, wd)
            return corr / torch.sqrt(torch.tensor(dim).float())


class MixedCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.corr_cs_pyramid = []

        # all pairs correlation
        corr, corr_cs = self.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        corr_cs = corr_cs.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        self.corr_cs_pyramid.append(corr_cs)

        for i in range(self.num_levels - 1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            corr, corr_cs = self.corr(fmap1, fmap2)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            corr_cs = corr_cs.reshape(batch * h1 * w1, dim, h2, w2)
            self.corr_pyramid.append(corr)
            self.corr_cs_pyramid.append(corr_cs)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        out_pyramid_cs = []

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            corr_cs = self.corr_cs_pyramid[i]

            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr_cs = bilinear_sampler(corr_cs, coords_lvl)

            corr = corr.view(batch, h1, w1, -1)
            corr_cs = corr_cs.view(batch, h1, w1, -1)

            out_pyramid.append(corr)
            out_pyramid_cs.append(corr_cs)

        out = torch.cat(out_pyramid, dim=-1)
        out_cs = torch.cat(out_pyramid_cs, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous().float(), out_cs.permute(0, 3, 1, 2).contiguous().float()

    def corr(self, fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        batch, dim, ht2, wd2 = fmap2.shape

        fmap1 = fmap1.view(batch, dim, ht1 * wd1)
        fmap2 = fmap2.view(batch, dim, ht2 * wd2)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)

        fmap1_mag = torch.linalg.vector_norm(fmap1, dim=1) + 1e-8
        fmap2_mag = torch.linalg.vector_norm(fmap2, dim=1) + 1e-8

        corr_cosine = corr / fmap1_mag.view(batch, -1, 1)
        corr_cosine = corr_cosine / fmap2_mag.view(batch, 1, -1)

        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2) / (dim**0.5)
        corr_cosine = corr_cosine.view(batch, ht1, wd1, 1, ht2, wd2)

        return corr, corr_cosine


class CosineCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = self.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)

        for i in range(self.num_levels - 1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            corr = self.corr(fmap1, fmap2)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def corr(self, fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        batch, dim, ht2, wd2 = fmap2.shape

        fmap1 = fmap1.view(batch, dim, ht1 * wd1)
        fmap2 = fmap2.view(batch, dim, ht2 * wd2)

        fmap1_mag = torch.linalg.vector_norm(fmap1, dim=1, keepdim=True) + 1e-8
        fmap2_mag = torch.linalg.vector_norm(fmap2, dim=1, keepdim=True) + 1e-8
        fmap1 = fmap1 / fmap1_mag
        fmap2 = fmap2 / fmap2_mag
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr


class CudaCorrBlock:
    def __init__(self, fmap1, fmap2, radius=4, dropout=1.0, num_levels=4,
                 lookup_softmax=False, lookup_softmax_all=False):
        self.dropout = dropout
        self.radius = radius
        self.num_levels = num_levels
        self.lookup_softmax = lookup_softmax
        self.lookup_softmax_all = lookup_softmax_all

        self.ii = torch.zeros(1, dtype=torch.long, device=fmap1.device)
        self.jj = torch.zeros(1, dtype=torch.long, device=fmap1.device)

        _, dim, _, _ = fmap1.shape

        self.scale = dim**0.5
        self.pyramid = [fmap2.unsqueeze(dim=1)]
        for i in range(self.num_levels - 1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append(fmap2.unsqueeze(dim=1))

        self.fmap1 = fmap1.unsqueeze(dim=1)

    def __call__(self, coords):
        b, _, h, w = coords.shape

        corrs = []
        for i in range(self.num_levels):
            corr_lvl = altcorr.corr(self.fmap1, self.pyramid[i], coords.unsqueeze(dim=1) / (2**i),
                                    self.ii, self.jj, self.radius, self.dropout) / self.scale

            corr_lvl = corr_lvl.reshape(b, -1, h, w)

            if self.lookup_softmax:
                corr_lvl = F.softmax(corr_lvl, dim=1)
            corrs.append(corr_lvl)

        corr = torch.cat(corrs, dim=1)

        if self.lookup_softmax_all:
            corr = F.softmax(corr, dim=1)

        return corr
