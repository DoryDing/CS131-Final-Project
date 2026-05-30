import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from diagram import draw_formation

PINK = "#FF2D78"
DARK_BG = "#1a1a1a"

def make_cover_page(title, formations):
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(DARK_BG)

    # big title in the center
    fig.text(0.5, 0.62, title,
             color="white", fontsize=28, fontweight="bold",
             ha="center", va="center")

    # pink underline under the title
    line = plt.Line2D([0.25, 0.75], [0.57, 0.57],
                      transform=fig.transFigure,
                      color=PINK, linewidth=2)
    fig.add_artist(line)

    # number of formations detected
    fig.text(0.5, 0.48, f"{len(formations)} formations detected",
             color="white", fontsize=14,
             ha="center", va="center")

    # timestamp range if available
    if formations and "start_time_s" in formations[0]:
        start = formations[0]["start_time_s"]
        end = formations[-1]["end_time_s"]
        fig.text(0.5, 0.40, f"{start}s  –  {end}s",
                 color="#aaaaaa", fontsize=11,
                 ha="center", va="center")

    # date generated at bottom
    today = datetime.date.today().strftime("%B %d, %Y")
    fig.text(0.5, 0.15, f"Generated {today}",
             color="#888888", fontsize=9,
             ha="center", va="center")

    return fig

def add_legend(fig, member_ids):
    # figure out where to place the legend — top left corner, above the stage
    # using figure coordinates (0 to 1)
    start_x = 0.04
    y = 0.91
    dot_spacing = 0.035

    for i, member_id in enumerate(sorted(member_ids)):
        x = start_x + i * dot_spacing

        # glow
        for radius, alpha in [(0.018, 0.06), (0.014, 0.12)]:
            circle = plt.Circle((x, y), radius,
                                color=PINK, alpha=alpha,
                                transform=fig.transFigure,
                                figure=fig, clip_on=False)
            fig.add_artist(circle)

        # solid dot
        dot = plt.Circle((x, y), 0.011,
                         color=PINK,
                         transform=fig.transFigure,
                         figure=fig, clip_on=False)
        fig.add_artist(dot)

        # number inside
        fig.text(x, y, str(member_id),
                 color="white", fontsize=7, fontweight="bold",
                 ha="center", va="center")


def add_page_number(fig, page_num):
    fig.text(0.96, 0.03, str(page_num),
             color="black", fontsize=10,
             ha="right", va="bottom")

def build_pdf(formations, output_path, title="Formation Sheet"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # collect all member ids across all formations for the legend
    all_member_ids = set()
    for formation in formations:
        all_member_ids.update(formation["positions"].keys())

    with PdfPages(output_path) as pdf:
        # page 1: cover
        cover = make_cover_page(title, formations)
        pdf.savefig(cover, facecolor=DARK_BG)
        plt.close(cover)

        # one page per formation
        for i, formation in enumerate(formations):
            fig = draw_formation(formation)
            add_legend(fig, all_member_ids)
            add_page_number(fig, i + 1)  # page 1 = first formation page
            pdf.savefig(fig, facecolor="white", bbox_inches="tight")
            plt.close(fig)

    print(f"saved PDF to {output_path}")
    return output_path

