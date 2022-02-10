from dataclasses import dataclass

@dataclass
class Metrics:
    N : int
    average_l2 : float
    final_l2 : float
    gt_col : float
    pred_col : float
    topk_ade : float
    topk_fde : float
    nll : float

    def __init__(self, N, average_l2, final_l2, gt_col, pred_col, topk_ade, topk_fde, nll):
        self.N = N
        self.average_l2 = average_l2
        self.final_l2 = final_l2
        self.gt_col = gt_col
        self.pred_col = pred_col
        self.topk_ade = topk_ade
        self.topk_fde = topk_fde
        self.nll = nll

    def __iadd__(self, other):
        self.N += other.N 
        self.average_l2 += other.average_l2 
        self.final_l2 += other.final_l2
        self.gt_col += other.gt_col
        if (other.pred_col == -1) or (self.pred_col == -1):
            self.pred_col = -1
        else:
            self.pred_col += other.pred_col
        self.topk_ade += other.topk_ade
        self.topk_fde += other.topk_fde
        self.nll += other.nll
        return self

    def avg_vals(self):
        if self.N == 0:
            return
        self.average_l2 /= self.N
        self.final_l2 /= self.N
        self.gt_col /= (0.01*self.N)
        if self.pred_col != -1:
            self.pred_col /= (0.01*self.N)
        self.topk_ade /= self.N
        self.topk_fde /= self.N
        self.nll /= self.N
    
    def to_list(self):
        return [self.N, self.average_l2, self.final_l2, self.pred_col, self.gt_col, self.topk_ade, self.topk_fde, self.nll]

    def avg_vals_to_list(self):
        self.avg_vals()
        return [self.N, self.average_l2, self.final_l2, self.pred_col, self.gt_col, self.topk_ade, self.topk_fde, self.nll]

@dataclass
class Categories:
    static_scenes : Metrics
    linear_scenes : Metrics
    forced_non_linear_scenes : Metrics
    non_linear_scenes : Metrics

    def __init__(self, static_scenes, linear_scenes, forced_non_linear_scenes, non_linear_scenes):
        self.static_scenes = static_scenes
        self.linear_scenes = linear_scenes
        self.forced_non_linear_scenes = forced_non_linear_scenes
        self.non_linear_scenes = non_linear_scenes

@dataclass
class Sub_categories:
    lf : Metrics
    ca : Metrics
    grp : Metrics
    others : Metrics

    def __init__(self, lf, ca, grp, others):
        self.lf = lf
        self.ca = ca
        self.grp = grp
        self.others = others

# out = Categories(*[Metrics(*([5] + [0.0]*8)) for i in range(1,5)])
# print(out)
# out1 = Metrics(*([5] + [1.0]*8))
# out2 = Metrics(*([7] + [1.0]*8))
# out1+=out2
# print(out1)

# out1 = Metrics(*([10] + [1.0]*8))
# print(out1.avg_vals())