"""Step 5: sequence networks over the same inputs and folds as the tree models.

    python -m src.idsse.train_nn cv gru      # leave-one-match-out on DEV; writes cache/idsse/cvpred_nn_gru.npz
    python -m src.idsse.train_nn cv tcn      # temporal convolutions
    python -m src.idsse.train_nn cv lstm     # LSTM variant of the GRU

It reads 5 Hz sequences (every 5th frame) in windows of WINDOW steps, looks both ways, and outputs
P(H, A, D) for every step; predictions are brought back to 25 Hz by repeating each step over its five
frames. Same training data as the trees (all mess levels, every window also mirrored home <-> away),
same validation (realistic level, prediction averaged with the mirror), same smoother and scoring.

Inputs are standardised with the training folds' medians and spreads; missing values become 0 after
standardising, with a per-input "missing" flag so the network can tell "absent" from "average".
Early stopping uses one of the four TRAINING matches, held out of the gradient steps; the validation
match is never seen before its score.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from . import dataset, decode, inputs, parse, splits

LEVELS = list(dataset.LEVELS)
SEED = 1
STEP = 5                 # 25 Hz -> 5 Hz
WINDOW = 150             # steps (30 s at 5 Hz)
HIDDEN = 96
EPOCHS = 12
BATCH = 256              # 4 x the first version with 1/4 of the steps: same windows per epoch, fewer launches
STEPS_PER_EPOCH = 100
LR = 2e-3
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
AMP = DEVICE == "cuda"   # mixed precision (float16) on NVIDIA GPUs; full precision elsewhere
if AMP:
    torch.backends.cudnn.benchmark = True
OUT = parse.CACHE_DIR


class Scaler:
    def fit(self, X):
        self.med = np.nanmedian(X, 0)
        q = np.nanpercentile(X, [10, 90], 0)
        self.scale = np.where(q[1] - q[0] > 1e-6, q[1] - q[0], 1.0)
        return self

    def transform(self, X):
        miss = ~np.isfinite(X)
        Z = np.clip((np.where(miss, self.med, X) - self.med) / self.scale, -10, 10)
        return np.concatenate([Z, miss], 1).astype(np.float32)


class RNNNet(nn.Module):
    """Bidirectional GRU or LSTM: reads the window forwards and backwards."""

    def __init__(self, n_in, cell="gru"):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Dropout(0.1))
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(HIDDEN, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * HIDDEN, 3)

    def forward(self, x):
        h, _ = self.rnn(self.inp(x))
        return self.out(h)


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation))
        self.norm = nn.GroupNorm(1, ch)

    def forward(self, x):
        return torch.nn.functional.gelu(self.norm(x + self.net(x)))


class TCNNet(nn.Module):
    """Temporal convolutions looking both ways at several time scales: dilations 1..32 steps give a
    reach of about +-25 s at 5 Hz."""

    def __init__(self, n_in, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Dropout(0.1))
        self.blocks = nn.Sequential(*[TCNBlock(HIDDEN, d) for d in dilations])
        self.out = nn.Linear(HIDDEN, 3)

    def forward(self, x):
        h = self.blocks(self.inp(x).transpose(1, 2)).transpose(1, 2)
        return self.out(h)


def make_net(arch, n_in):
    return TCNNet(n_in) if arch == "tcn" else RNNNet(n_in, cell=arch)


def sequences(a, key="X", ykey="y"):
    """Per half: 5 Hz inputs and labels (the label of the step's first frame)."""
    out = []
    for h in np.unique(a["half"]):
        idx = np.flatnonzero(a["half"] == h)[::STEP]
        out.append((a[key][idx], a[ykey][idx]))
    return out


def windows(seqs, rng):
    xs, ys = [], []
    for X, y in seqs:
        off = rng.integers(WINDOW)
        for s in range(off, len(X) - WINDOW, WINDOW // 2):
            xs.append(X[s:s + WINDOW])
            ys.append(y[s:s + WINDOW])
    return np.stack(xs), np.stack(ys)


def predict_seq(net, X):
    net.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
        logits = net(torch.from_numpy(X[None]).to(DEVICE))[0].float()
    return torch.softmax(logits, -1).cpu().numpy()


def predict_match(net, scaler, a):
    """P(H, A, D) at 25 Hz for a whole match, averaged with the mirrored prediction."""
    p = np.zeros((len(a["y"]), 3), np.float32)
    for h in np.unique(a["half"]):
        idx = np.flatnonzero(a["half"] == h)
        step_idx = idx[::STEP]
        pa = predict_seq(net, scaler.transform(a["X"][step_idx]))
        pm = predict_seq(net, scaler.transform(a["Xm"][step_idx]))[:, [1, 0, 2]]
        ps = (pa + pm) / 2
        p[idx] = np.repeat(ps, STEP, 0)[:len(idx)]
    return p


def train_fold(train, rng, arch="gru"):
    stop_match, fit_matches = train[0], train[1:]
    raw = [inputs.get(m, l, SEED) for m in fit_matches for l in LEVELS]
    scaler = Scaler().fit(np.concatenate([a["X"][::25] for a in raw]))
    seqs = []
    for a in raw:
        for key, ykey in (("X", "y"), ("Xm", "ym")):
            seqs += [(scaler.transform(X), y) for X, y in sequences(a, key, ykey)]
    del raw
    stop = inputs.get(stop_match, "realistic", SEED)
    net = make_net(arch, seqs[0][0].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * STEPS_PER_EPOCH)
    lossf = nn.CrossEntropyLoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        Xw, yw = windows(seqs, rng)
        order = rng.permutation(len(Xw))
        for b, s in enumerate(range(0, len(order), BATCH)):
            if b >= STEPS_PER_EPOCH:
                break
            j = order[s:s + BATCH]
            x = torch.from_numpy(Xw[j]).to(DEVICE, non_blocking=True)
            y = torch.from_numpy(yw[j].astype(np.int64)).to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                loss = lossf(net(x).reshape(-1, 3).float(), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            sched.step()
        p = predict_match(net, scaler, stop)
        ll = float(-np.mean(np.log(np.clip(p[np.arange(len(p)), stop["y"]], 1e-6, 1))))
        if ll < best:
            best, best_state = ll, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print("    epoch {} early-stop logloss {:.4f}".format(ep + 1, ll), flush=True)
    net.load_state_dict(best_state)
    return net, scaler


def cv(arch="gru"):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    folds = splits.cv_folds()
    preds, cases = {}, {}
    t0 = time.time()
    for train, val in folds:
        print("[{}] fold -> {} (early stop on {})".format(arch, val, train[0]), flush=True)
        net, scaler = train_fold(train, rng, arch)
        a = inputs.get(val, "realistic", SEED)
        preds[val] = predict_match(net, scaler, a)
        cases[val] = (np.log(np.clip(preds[val], 1e-6, 1)), a["half"], a["y"])
        print("  done ({:.0f} s)".format(time.time() - t0), flush=True)
    np.savez(os.path.join(OUT, "cvpred_nn_{}.npz".format(arch)), **preds)
    res = decode.cv_scores(decode.grid_scores(cases), folds)
    print(res.round(3).to_string(index=False))
    print("[{}] mean spellF1 {:.3f} | macroF1 {:.3f}".format(arch, res.spellF1.mean(), res.macroF1.mean()))


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv(sys.argv[2] if len(sys.argv) > 2 else "gru")      # gru | lstm | tcn
