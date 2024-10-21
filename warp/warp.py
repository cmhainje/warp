import jax
import jax.numpy as jnp

from scipy.spatial import KDTree
from optax import adam, apply_updates
from tqdm.auto import tqdm

from .fourier import basis_indices, basis_2d


class Warp():
    def __init__(self,
        orders=(3, 3), scales=(None, None), theta_0=None,
        k_ngb=200, cap=10, alpha=0.1,
        max_iter=120, lr=None, auto_refine=True, quiet=False
    ):
        self.orders_ = orders
        self.n_ngb_ = k_ngb
        self.cap_ = cap
        self.alpha_ = alpha
        self.scales_ = scales
        self.max_iter_ = max_iter
        if lr is None:
            def _default_schedule(epoch: int):
                lr_0, lr_1 = 5e-1, 5e-2
                ep_0, ep_1 = 25, 100

                if epoch < ep_0:
                    return lr_0
                if epoch < ep_1:
                    t = (epoch - ep_0) / (ep_1 - ep_0)
                    return lr_0 * (1 - t) + lr_1 * t
                return lr_1

            self.lr_schedule_ = _default_schedule
        elif callable(lr):
            self.lr_schedule_ = lr
        else:
            self.lr_schedule_ = lambda epoch: lr
        self.auto_refine_ = auto_refine
        self.quiet_ = quiet

        self.loss_ = None
        self.loss_curve_ = []
        self.n_iter_ = 0
        self.refined_ = False

        self.thetas_ = []

        if theta_0 is not None:
            self.coefs_ = theta_0
        else:
            self.coefs_ = jnp.zeros((2, (2 * orders[0] + 1) * (2 * orders[1] + 1)))

        self.ngb_ = None

        self.indices_ = basis_indices(orders)

    def __repr__(self):
        return f'Warp(orders={self.orders_}, k_ngb={self.n_ngb_}, cap={self.cap_}, alpha={self.alpha_}, scales={self.scales_}, max_iter={self.max_iter_}, auto_refine={self.auto_refine_})'

    def _update_model(self, orders, scales):
        self.orders_ = orders
        self.scales_ = scales
        self.indices_ = basis_indices(orders)

    def _basis(self, control):
        return basis_2d(
            self.indices_[:, 0], self.indices_[:, 1],
            control[:, 0], control[:, 1],
            self.scales_[0], self.scales_[1]
        )

    def _distort_factory(self, control):
        basis_eval = self._basis(control)

        @jax.jit
        def distort(theta):
            a, b = theta
            dx = (a[:, None] * basis_eval).sum(axis=0)
            dy = (b[:, None] * basis_eval).sum(axis=0)
            return control + jnp.stack([dx, dy], axis=-1)

        return distort

    def fit(self, control, target, weights_c=None, weights_t=None):

        if weights_c is None:
            weights_c = jnp.ones(control.shape[0])
        if weights_t is None:
            weights_t = jnp.ones(target.shape[0])

        distort = self._distort_factory(control)

        def find_target_neighbors(k):
            tree = KDTree(target)
            model_0 = distort(self.coefs_)
            dists, idx = tree.query(model_0, k=k)
            return jnp.array(idx)

        idx_t = find_target_neighbors(self.n_ngb_)

        def find_control_neighbors(k):
            model_0 = distort(self.coefs_)
            tree = KDTree(model_0)
            dists, idx = tree.query(target, k=k)
            return jnp.array(idx)

        idx_c = find_control_neighbors(self.n_ngb_)

        @jax.jit
        def delta_sqs_t(model):
            return jnp.square(model[None, idx_c] - target[:, None]).sum(-1).min(-1).squeeze()

        @jax.jit
        def delta_sqs_c(model):
            return jnp.square(model[:, None] - target[None, idx_t]).sum(-1).min(-1).squeeze()

        @jax.jit
        def cmse(dsq):
            return 1. / (1. / dsq + 1. / self.cap_)

        @jax.jit
        def l2_regularization(theta):
            return jnp.square(theta).mean()

        @jax.jit
        def forward(theta):
            d = distort(theta)
            return (
                jnp.average(cmse(delta_sqs_c(d)), weights=weights_c)
                + jnp.average(cmse(delta_sqs_t(d)), weights=weights_t)
                + self.alpha_ * l2_regularization(theta)
            )

        forward_grad = jax.value_and_grad(forward)

        def lr_schedule(epoch: int):
            lr_0, lr_1 = 5e-1, 5e-2
            ep_0, ep_1 = 25, 100

            if epoch < ep_0:
                return lr_0
            if epoch < ep_1:
                t = (epoch - ep_0) / (ep_1 - ep_0)
                return lr_0 * (1 - t) + lr_1 * t
            return lr_1

        opt = adam(learning_rate=self.lr_schedule_)
        opt_state = opt.init(self.coefs_)

        _iter = range(self.max_iter_)
        if not self.quiet_:
            _iter = tqdm(_iter, desc="fitting", unit="epoch", total=self.max_iter_)

        self.thetas_.append(self.coefs_)

        for epoch in _iter:
            loss_value, grads = forward_grad(self.coefs_)

            if jnp.isnan(loss_value).any() or jnp.isnan(grads).any():
                raise RuntimeError("NaN encountered, stopping early")

            updates, opt_state = opt.update(grads, opt_state)
            self.coefs_ = apply_updates(self.coefs_, updates)

            self.loss_curve_.append(loss_value)
            self.loss_ = loss_value
            self.n_iter_ += 1
            self.thetas_.append(self.coefs_)

        if self.auto_refine_:
            self.refine(control, target)

        return self

    def refine(self, control, target):
        self.refined_ = True

        # form one-to-one map
        distort = self._distort_factory(control)
        model = distort(self.coefs_)

        tree = KDTree(model)
        dists, indices = tree.query(target, k=1)
        dists = jnp.array(dists)

        mask = dists > 10**2  # ten pixels
        indices[mask] = -1

        with jax.default_device(jax.devices("cpu")[0]):

            # de-duplicate indices: among the targets whose closest control is i, keep only the one which is closest to i
            for i in range(len(model)):
                mask = indices == i
                num = jnp.count_nonzero(mask)

                if num > 1:
                    # find the min distance
                    _d = dists.copy()
                    _d = _d.at[~mask].set(jnp.nan)
                    # _d[~mask] = jnp.nan  # nan out all the other distances
                    j = jnp.nanargmin(_d)  # find the index of the min non-nan value

                    # kill the indices of the others
                    mask[j] = False
                    indices[mask] = -1

        # cut to subset and re-order controls/models to match targets
        target = target[indices != -1]
        model = model[indices[indices != -1]]
        control = control[indices[indices != -1]]

        # fit the subset with lstsq
        decision = self._basis(control).T
        resid = target - model
        delta_theta = jnp.linalg.lstsq(decision, resid)[0].T
        self.coefs_ = self.coefs_ + delta_theta

        # would the following be better?
        # self.coefs_ = jnp.linalg.lstsq(decision, target - control)[0]

        return self

    def transform(self, X):
        distort = self._distort_factory(X)
        return distort(self.coefs_)
