"""Top-down animations of single moments from IDSSE tracking, for reviewing labels without video.

    from src.idsse import anim
    anim.render_moment(m, row, "moment_01.gif", title="#1  Did Köln gain control?", colors=..., names=...)
    anim.write_index(folder, sheet)   # index.html showing every GIF with its question

Each GIF shows +-window_s around the moment at reduced speed: players as dots in team colours, the
ball with a short trail, the match clock, and notes when the ball is in the air or out of play. It shows
no possession label of any kind, so a review stays blind.
"""

import html
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from . import parse  # noqa: E402

FPS = parse.FPS
L, W = 105.0, 68.0


def kickoff_rows(m):
    """First live frame of each half (BallStatus alive)."""
    return {h: int(np.flatnonzero((m["half"] == h) & (m["status"] == 1))[0]) for h in np.unique(m["half"])}


def clock(m, row, kick=None):
    kick = kick or kickoff_rows(m)
    h = m["half"][row]
    s = (row - kick[h]) / FPS + (45 * 60 if h == 2 else 0)
    return "{:02d}:{:02d}".format(int(s // 60), int(s % 60))


def draw_pitch(ax):
    kw = dict(color="white", lw=1.2, zorder=1)
    ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, fill=False, **kw))
    ax.plot([0, 0], [-W / 2, W / 2], **kw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * L / 2 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * L / 2 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * L / 2 - (0 if s > 0 else 2), -3.66), 2, 7.32, fill=False, **kw))
        ax.add_patch(Arc((s * (L / 2 - 11), 0), 18.3, 18.3, theta1=(127 if s > 0 else -53),
                         theta2=(233 if s > 0 else 53), **kw))
    ax.set_xlim(-L / 2 - 4, L / 2 + 4)
    ax.set_ylim(-W / 2 - 4, W / 2 + 4)
    ax.set_aspect("equal")
    ax.set_facecolor("#3a7d44")
    ax.set_xticks([]), ax.set_yticks([])


def render_moment(m, row, path, title, colors, names, window_s=5.0, step=2, play_fps=8, kick=None):
    """GIF of rows [row - window_s, row + window_s] inside the moment's half. step=2 at play_fps=8 is
    about 0.64x real speed."""
    kick = kick or kickoff_rows(m)
    in_half = np.flatnonzero(m["half"] == m["half"][row])
    lo = max(row - int(window_s * FPS), in_half[0])
    hi = min(row + int(window_s * FPS), in_half[-1])
    rows = np.arange(lo, hi + 1, step)
    team = m["person_team"]

    # which goal each team defends in this half, from where its keeper stands
    goal_side = {}
    for t in (0, 1):
        j = np.flatnonzero(m["person_gk"] & (team == t))
        x = np.nanmean(m["xy"][in_half][:, j, 0]) if len(j) else np.nan
        goal_side[t] = np.sign(x) if np.isfinite(x) else (-1 if t == 0 else 1)

    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=80)
    draw_pitch(ax)
    fig.suptitle(title, fontsize=11)
    for t in (0, 1):
        ax.text(goal_side[t] * (L / 2 + 2), W / 2 + 2.2, names[t] + " goal", ha="center",
                fontsize=8, color=colors[t], fontweight="bold")
    dots = {t: ax.scatter([], [], s=70, c=colors[t], edgecolors="white", linewidths=0.8, zorder=3)
            for t in (0, 1)}
    trail, = ax.plot([], [], color="yellow", lw=1.5, alpha=0.8, zorder=4)
    ball = ax.scatter([], [], s=45, c="white", edgecolors="black", linewidths=1.2, zorder=5)
    info = ax.text(-L / 2, -W / 2 - 3.2, "", fontsize=9, color="black", va="center")
    flag = ax.text(L / 2, -W / 2 - 3.2, "", fontsize=9, color="darkred", ha="right", va="center",
                   fontweight="bold")

    def frame(r):
        for t in (0, 1):
            xy = m["xy"][r][team == t]
            dots[t].set_offsets(xy[np.isfinite(xy[:, 0])])
        tr = m["ball"][max(r - FPS, lo):r + 1, :2]
        trail.set_data(tr[:, 0], tr[:, 1])
        ball.set_offsets(m["ball"][r:r + 1, :2])
        dt = (r - row) / FPS
        info.set_text("{}   moment {:+.1f} s".format(clock(m, r, kick), dt))
        notes = []
        if abs(dt) <= 0.5:
            notes.append("<< MOMENT >>")
        if m["ball"][r, 2] > 1.5:
            notes.append("ball in the air ({:.1f} m)".format(m["ball"][r, 2]))
        if m["status"][r] == 0:
            notes.append("ball out of play")
        flag.set_text("   ".join(notes))
        return [*dots.values(), trail, ball, info, flag]

    FuncAnimation(fig, frame, frames=rows, blit=True).save(path, writer=PillowWriter(fps=play_fps))
    plt.close(fig)


def write_index(folder, sheet, gif_name="moment_{:02d}.gif", title="Review"):
    """index.html in folder: every moment's GIF with its id, clock, question and location."""
    parts = ["<!doctype html><meta charset='utf-8'><title>{}</title>".format(html.escape(title)),
             "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}"
             "figure{margin:0 0 32px}img{max-width:100%}</style>",
             "<h1>{}</h1><p>Answer per moment: <b>Y</b> gained control · <b>T</b> touch / deflection only · "
             "<b>P</b> ball went past / normal pass of the other team · <b>?</b> can't tell.</p>"
             .format(html.escape(title))]
    for r in sheet.itertuples(index=False):
        parts.append("<figure><h3>#{} &nbsp; half {} &nbsp; {}</h3><p>{} <i>({})</i></p>"
                     "<img src='{}'></figure>".format(r.id, r.half, r.clock, html.escape(r.question),
                                                      html.escape(str(r.where)), gif_name.format(r.id)))
    path = os.path.join(folder, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path
