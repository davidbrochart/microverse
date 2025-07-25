import numpy as np


def run_gr4j(x, p, e, q, s, uh1_array, uh2_array, l, m):
    for t in range(p.size):
        if p[t] > e[t]:
            pn = p[t] - e[t]
            en = 0.
            tmp = s[0] / x[0]
            ps = x[0] * (1. - tmp ** 2) * np.tanh(pn / x[0]) / (1. + tmp * np.tanh(pn / x[0]))
            s[0] += ps
        elif p[t] < e[t]:
            ps = 0.
            pn = 0.
            en = e[t] - p[t]
            tmp = s[0] / x[0]
            es = s[0] * (2. - tmp) * np.tanh(en / x[0]) / (1. + (1. - tmp) * np.tanh(en / x[0]))
            tmp = s[0] - es
            if tmp > 0.:
                s[0] = tmp
            else:
                s[0] = 0.
        else:
            pn = 0.
            en = 0.
            ps = 0.
        tmp = (4. * s[0] / (9. * x[0]))
        perc = s[0] * (1. - (1. + tmp ** 4) ** (-1. / 4.))
        s[0] -= perc
        pr_0 = perc + pn - ps
        q9 = 0.
        q1 = 0.
        for i in range(m):
            if i == 0:
                pr_i = pr_0
            else:
                pr_i = s[2 + i - 1]
            if i < l:
                q9 += uh1_array[i] * pr_i
            q1 += uh2_array[i] * pr_i
        q9 *= 0.9
        q1 *= 0.1
        f = x[1] * ((s[1] / x[2]) ** (7. / 2.))
        tmp = s[1] + q9 + f
        if tmp > 0.:
            s[1] = tmp
        else:
            s[1] = 0.
        tmp = s[1] / x[2]
        qr = s[1] * (1. - ((1. + tmp ** 4) ** (-1. / 4.)))
        s[1] -= qr
        tmp = q1 + f
        if tmp > 0.:
            qd = tmp
        else:
            qd = 0.
        q[t] = qr + qd
        if s.size > 2:
            s[3:] = s[2:-1]
            s[2] = pr_0


class GR4J:

    def __init__(self, x):
        self.x = np.array(x)
        self.s = np.empty(2 + int(2. * self.x[3]))
        self.s[0] = self.x[0] / 2.
        self.s[1] = self.x[2] / 2.
        self.s[2:] = 0.
        self.l = int(self.x[3]) + 1
        self.m = int(2. * self.x[3]) + 1
        self.uh1_array = np.empty(self.l)
        self.uh2_array = np.empty(self.m)
        for i in range(self.m):
            if i < self.l:
                self.uh1_array[i] = self.uh1(i + 1)
            self.uh2_array[i] = self.uh2(i + 1)

    def sh1(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = (float(t) / self.x[3]) ** (5. / 2.)
        else:
            res = 1.
        return res

    def sh2(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = 0.5 * ((float(t) / self.x[3]) ** (5. / 2.))
        elif t < 2. * self.x[3]:
            res = 1. - 0.5 * ((2. - float(t) / self.x[3]) ** (5. / 2.))
        else:
            res = 1.
        return res

    def uh1(self, j):
        return self.sh1(j) - self.sh1(j - 1)

    def uh2(self, j):
        return self.sh2(j) - self.sh2(j - 1)

    def run(self, pe):
        q = np.empty_like(pe[0])
        run_gr4j(self.x, pe[0], pe[1], q, self.s, self.uh1_array, self.uh2_array, self.l, self.m)
        return [q]


def calibration(x, in_obs, out_obs, warmup_period, crit_func, model, x_range=None, x_fix=None):
    if x_range is None:
        _x = x
    else:
        _x = []
        for i in range(len(x_range)):
            if x_fix is None or x_fix[i] is None:
                if x[i] < x_range[i][0]:
                    return np.inf
                if x[i] > x_range[i][1]:
                    return np.inf
                _x.append(x[i])
            else:
                _x.append(x_fix[i])
    q_mod = model(_x)
    out_sim = q_mod.run(in_obs)
    error = crit_func(out_obs[0][warmup_period:], out_sim[0][warmup_period:])
    return error


def nse(x_obs, x_est):
    _x_obs = x_obs[~np.isnan(x_obs)]
    _x_est = x_est[~np.isnan(x_obs)]
    return 1. - (np.sum(np.square(_x_obs - _x_est)) / np.sum(np.square(_x_obs - np.mean(_x_obs))))


def nse_min(x_obs, x_est):
    return 1. - nse(x_obs, x_est)


# original code at: https://github.com/olivierverdier/downhill-simplex/blob/19c2c5b25ee6c7ae5bb231e90db168e389dbe8b8/downhill_simplex.py

def generate_simplex(x0, step=0.1):
    """
    Create a simplex based at x0
    """
    yield x0.copy()
    for i,_ in enumerate(x0):
        x = x0.copy()
        x[i] += step
        yield x

def make_simplex(x0, step=0.1):
    return np.array(list(generate_simplex(x0, step)))

def centroid(points):
    """
    Compute the centroid of a list points given as an array.
    """
    return np.mean(points, axis=0)


class DownhillSimplex(object):

    refl = 1.
    ext = 1.
    cont = 0.5
    red = 0.5

    # max_stagnations: break after max_stagnations iterations with an improvement lower than no_improv_thr
    no_improve_thr=10e-6
    max_stagnations=10

    max_iter=1000
    
    def __init__(self, f, points):
        '''
            f: (function): function to optimize, must return a scalar score 
                and operate over a numpy array of the same dimensions as x_start
            points: (numpy array): initial position
        '''
        self.f = f
        self.points = points

    def step(self, res):
        # centroid of the lowest face
        pts = np.array([tup[0] for tup in res[:-1]])
        x0 = centroid(pts)

        new_res = self.reflection(res, x0, self.refl)
        if new_res is not None:
            exp_res = self.expansion(new_res, x0, self.ext)
            if exp_res is not None:
                new_res = exp_res
        else:
            new_res = self.contraction(res, x0, self.cont)
            if new_res is None:
                new_res = self.reduction(res, self.red)
        return new_res

    def run(self):
        # initialize
        self.prev_best = self.f(self.points[0])
        self.stagnations = 0
        res = self.make_score(self.points)

        # simplex iter
        for iters in range(self.max_iter):
            res = self.sort(res)
            best = res[0][1]

            # break after max_stagnations iterations with no improvement
            if best < self.prev_best - self.no_improve_thr:
                self.stagnations = 0
                self.prev_best = best
            else:
                self.stagnations += 1
        
            if self.stagnations >= self.max_stagnations:
                return res[0]

            # Downhill-Simplex algorithm
            new_res = self.step(res)

            res = new_res
        else:
            raise Exception("No convergence after {} iterations".format(iters))


    def sort(self, res):
        """
        Order the points according to their value.
        """
        return sorted(res, key = lambda x: x[1])

    def reflection(self, res, x0, refl):
        """
        Reflection-extension step.
        refl: refl = 1 is a standard reflection
        """
        # reflected point and score
        xr = x0 + refl*(x0 - res[-1][0])
        rscore = self.f(xr)

        new_res = res[:]

        progress = rscore < new_res[-2][1]
        if progress: # if this is a progress, we keep it
            new_res[-1] = (xr, rscore)
            return new_res
        return None

    def expansion(self, res, x0, ext):
        """
        ext: the amount of the expansion; ext=0 means no expansion
        """
        xr, rscore = res[-1]
        # if it is the new best point, we try to expand
        if rscore < res[0][1]:
            xe = xr + ext*(xr - x0)
            escore = self.f(xe)
            if escore < rscore:
                new_res = res[:]
                new_res[-1] = (xe, escore)
                return new_res
        return None

    def contraction(self, res, x0, cont):
        """
        cont: contraction parameter: should be between zero and one
        """
        xc = x0 + cont*(res[-1][0] - x0)
        cscore = self.f(xc)

        new_res = res[:]

        progress = cscore < new_res[-1][1]
        if progress:
            new_res[-1] = (xc, cscore)
            return new_res
        return None

    def reduction(self, res, red):
        """
        red: reduction parameter: should be between zero and one
        """
        pts = np.array([pts for (pts,_) in res])
        dirs = pts - pts[0]
        reduced_points = pts[0] + red*dirs
        new_res = self.make_score(reduced_points)
        return new_res

    def make_score(self, points):
        res = [(pt, self.f(pt)) for pt in points]
        return res
