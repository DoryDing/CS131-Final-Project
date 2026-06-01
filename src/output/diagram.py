import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json

PINK = "#FF2D78"
DARK_BG = "#1a1a1a"
GRID_COLOR = "#2e2e2e"

def draw_formation(formation, stage_w=100, stage_h=100):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")  # page background is white

    # dark stage
    stage = patches.FancyBboxPatch(
        (0, 0), stage_w, stage_h,
        boxstyle="round,pad=0,rounding_size=4",
        linewidth=2.5,
        edgecolor=PINK,
        facecolor=DARK_BG,
        zorder=1
    )
    ax.add_patch(stage)

    # grid lines inside the stage
    # draw vertical lines
    for x in range(0, stage_w + 1, 10):
        ax.plot([x, x], [0, stage_h], color=GRID_COLOR, linewidth=0.5, zorder=2)
    # draw horizontal lines
    for y in range(0, stage_h + 1, 10):
        ax.plot([0, stage_w], [y, y], color=GRID_COLOR, linewidth=0.5, zorder=2)

    # center reference tick marks on each edge
    tick_len = 3
    cx = stage_w / 2
    cy = stage_h / 2
    ax.plot([cx, cx], [0, tick_len], color=PINK, linewidth=2, zorder=3)          # top
    ax.plot([cx, cx], [stage_h - tick_len, stage_h], color=PINK, linewidth=2, zorder=3)  # bottom
    ax.plot([0, tick_len], [cy, cy], color=PINK, linewidth=2, zorder=3)          # left
    ax.plot([stage_w - tick_len, stage_w], [cy, cy], color=PINK, linewidth=2, zorder=3)  # right

    # center X marker
    x_size = 3
    ax.plot([cx - x_size, cx + x_size], [cy - x_size, cy + x_size], color=PINK, linewidth=1.5, zorder=3)
    ax.plot([cx - x_size, cx + x_size], [cy + x_size, cy - x_size], color=PINK, linewidth=1.5, zorder=3)

    # member dots
    for member_id, pos in formation["positions"].items():
        x = pos["x"]
        y = pos["y"]

        for radius, alpha in [(5.5, 0.06), (4.5, 0.1), (3.5, 0.18)]:
            glow = plt.Circle((x, y), radius, color=PINK, alpha=alpha, zorder=4)
            ax.add_patch(glow)

        dot = plt.Circle((x, y), 2.8, color=PINK, zorder=5)
        ax.add_patch(dot)

        # member number in white inside the dot
        ax.text(x, y, str(member_id),
                color="white", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=6)

    # BACKSTAGE and AUDIENCE labels
    ax.text(cx, -4, "BACKSTAGE", color="black", fontsize=9,
            ha="center", va="center", fontfamily="monospace")
    ax.text(cx, stage_h + 4, "AUDIENCE", color="black", fontsize=9,
            ha="center", va="center", fontfamily="monospace")

    # title
    ax.set_title(
        f"Formation {formation['formation_id']}",
        fontsize=16, fontweight="bold", color="black",
        pad=20
    )
    # draw underline manually under title
    ax.axhline(y=stage_h + 8, xmin=0.3, xmax=0.7, color="black", linewidth=1.5, clip_on=False)

    # frame range + timestamp — placed below the AUDIENCE label
    start_f = formation.get("start_frame", "?")
    end_f   = formation.get("end_frame",   "?")
    frame_str = f"frames {start_f} – {end_f}"
    if "start_time_s" in formation:
        frame_str += f"  ({formation['start_time_s']}s – {formation['end_time_s']}s)"
    ax.text(cx, stage_h + 12, frame_str,
            color="#888888", fontsize=7,
            ha="center", va="center",
            clip_on=False)

    # axis settings
    ax.set_xlim(-8, stage_w + 8)
    ax.set_ylim(-8, stage_h + 8)
    ax.invert_yaxis()   # y=0 at top like video coords
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    return fig

def render_all_formations(formations, output_dir, stage_w=100, stage_h=100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for formation in formations:
        fig = draw_formation(formation, stage_w, stage_h)
        out_path = output_dir / f"formation_{formation['formation_id']}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(out_path)
        print(f"saved {out_path}")

    return saved