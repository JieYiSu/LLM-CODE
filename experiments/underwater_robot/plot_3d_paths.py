"""Clean publication plots for the underwater path-planning experiments.

The earlier figure showed both mean and best trajectories for every method and
used a dense 3-D current-vector grid. This version keeps one best path per
method, a sparse current-field cue, compact obstacle geometry, and a consistent
view across all scenarios.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .current_fields import CurrentField
    from .underwater_scenarios import WORKSPACE_HIGH, WORKSPACE_LOW, get_scenario
except ImportError:  # direct execution from the repository root
    import sys

    _PROJECT = Path(__file__).resolve().parents[2]
    if str(_PROJECT) not in sys.path:
        sys.path.insert(0, str(_PROJECT))
    from experiments.underwater_robot.current_fields import CurrentField
    from experiments.underwater_robot.underwater_scenarios import (
        WORKSPACE_HIGH,
        WORKSPACE_LOW,
        get_scenario,
    )


METHOD_ORDER = (
    "LLMOperatorDE",
    "TSDDEO",
    "ESA",
    "SADE-SS",
    "SGDLCO",
    "BGP-SAEA",
)

# Keep the internal result directory name stable while using the final
# manuscript name consistently in all exported figures.
METHOD_LABELS = {
    "LLMOperatorDE": "LLM-CODE",
}

METHOD_COLORS = {
    "LLMOperatorDE": "#009E73",
    "TSDDEO": "#D55E00",
    "ESA": "#0072B2",
    "SADE-SS": "#CC79A7",
    "SGDLCO": "#E69F00",
    "BGP-SAEA": "#56B4E9",
}

SCENARIO_TITLES = {
    "uniform_cross_current": "Uniform cross-current",
    "shear_vortex": "Shear-vortex current",
    "time_varying_multi_obstacle": "Time-varying multi-obstacle field",
}


def _load_paths(root: Path) -> dict[str, list[tuple[np.ndarray, float]]]:
    """Load all saved trajectories grouped by algorithm."""

    paths: dict[str, list[tuple[np.ndarray, float]]] = {}
    for file in sorted(root.glob("*/seed_*/best_trajectory.npz")):
        try:
            with np.load(file) as payload:
                path = np.asarray(payload["path"], dtype=float)
                score = float(payload["objective"]) if "objective" in payload else float("inf")
        except (OSError, KeyError, ValueError):
            continue
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            continue
        paths.setdefault(file.parent.parent.name, []).append((path, score))
    return paths


def _depth_path(path: np.ndarray) -> np.ndarray:
    """Convert the model's negative z coordinate to positive downward depth."""

    result = np.asarray(path, dtype=float).copy()
    result[:, 2] *= -1.0
    return result


def _box_faces(center: np.ndarray, half_size: np.ndarray) -> list[list[np.ndarray]]:
    corners = np.array(
        [
            center + np.array([sx * half_size[0], sy * half_size[1], sz * half_size[2]])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    return [
        [corners[i] for i in (0, 1, 3, 2)],
        [corners[i] for i in (4, 5, 7, 6)],
        [corners[i] for i in (0, 1, 5, 4)],
        [corners[i] for i in (2, 3, 7, 6)],
        [corners[i] for i in (0, 2, 6, 4)],
        [corners[i] for i in (1, 3, 7, 5)],
    ]


def _draw_obstacles(axis, obstacles: list[dict]) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    edge = "#B84A4A"
    face = "#D96B6B"
    for obstacle in obstacles:
        center = np.asarray(obstacle["center"], dtype=float)
        center[2] *= -1.0
        if obstacle["type"] == "box":
            half = np.asarray(obstacle["half_size"], dtype=float)
            half[2] = abs(half[2])
            axis.add_collection3d(
                Poly3DCollection(
                    _box_faces(center, half),
                    alpha=0.14,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.5,
                )
            )
        else:
            theta = np.linspace(0.0, 2.0 * np.pi, 36)
            depth = np.linspace(
                center[2] - float(obstacle["half_height"]),
                center[2] + float(obstacle["half_height"]),
                2,
            )
            theta_grid, depth_grid = np.meshgrid(theta, depth)
            x_grid = center[0] + float(obstacle["radius"]) * np.cos(theta_grid)
            y_grid = center[1] + float(obstacle["radius"]) * np.sin(theta_grid)
            axis.plot_surface(
                x_grid,
                y_grid,
                depth_grid,
                alpha=0.14,
                color=face,
                edgecolor=edge,
                linewidth=0.18,
                shade=True,
            )


def _draw_current(axis, scenario_data: dict) -> None:
    """Draw a sparse, low-opacity current cue instead of a dense arrow grid."""

    field = CurrentField(scenario_data["current"])
    xy = np.linspace(-1.25, 1.25, 3)
    depths = np.asarray([1.1, 2.6, 3.8], dtype=float)
    gx, gy, gd = np.meshgrid(xy, xy, depths, indexing="ij")
    points = np.column_stack((gx.ravel(), gy.ravel(), -gd.ravel()))
    flow = np.asarray(field.velocity(points, 0.0), dtype=float)
    flow[:, 2] *= -1.0
    axis.quiver(
        points[:, 0],
        points[:, 1],
        -points[:, 2],
        flow[:, 0],
        flow[:, 1],
        flow[:, 2],
        length=0.18,
        normalize=True,
        arrow_length_ratio=0.28,
        color="#56C5D8",
        alpha=0.25,
        linewidth=0.45,
    )


def _draw_workspace_box(axis) -> None:
    """Draw all twelve edges of the 3-D search workspace.

    Matplotlib's 3-D panes do not consistently expose all four sides of the
    projected x/y frame in vector exports. Drawing the box explicitly keeps
    the upper, lower, left and right workspace boundaries visible in PDF/SVG.
    """

    x0, x1 = float(WORKSPACE_LOW[0]), float(WORKSPACE_HIGH[0])
    y0, y1 = float(WORKSPACE_LOW[1]), float(WORKSPACE_HIGH[1])
    z0, z1 = float(-WORKSPACE_HIGH[2]), float(-WORKSPACE_LOW[2])
    edge_color = "#A6B1B9"
    edge_style = {
        "color": edge_color,
        "linewidth": 0.65,
        "alpha": 0.72,
        "solid_capstyle": "round",
        "zorder": 1,
    }

    # Four x-directed edges: lower/upper and front/back.
    for y in (y0, y1):
        for z in (z0, z1):
            axis.plot([x0, x1], [y, y], [z, z], **edge_style)
    # Four y-directed edges.
    for x in (x0, x1):
        for z in (z0, z1):
            axis.plot([x, x], [y0, y1], [z, z], **edge_style)
    # Four depth-directed edges.
    for x in (x0, x1):
        for y in (y0, y1):
            axis.plot([x, x], [y, y], [z0, z1], **edge_style)


def plot_3d_paths(root: str | Path, scenario: str, output: str | Path) -> Path:
    """Create a clean 3-D comparison plot and vector exports.

    Each method contributes one path: the lowest-objective trajectory across
    all saved seeds. The input result files and numerical experiments are not
    changed.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    root = Path(root)
    output = Path(output)
    scenario_data = get_scenario(scenario)
    method_paths = _load_paths(root)
    if not method_paths:
        raise FileNotFoundError(f"no best_trajectory.npz files found below {root}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.6, 4.8), dpi=220, facecolor="white")
    axis = fig.add_subplot(111, projection="3d")
    axis.set_facecolor("#FBFCFD")
    axis.set_proj_type("ortho")

    axis.set_xlim(float(WORKSPACE_LOW[0]), float(WORKSPACE_HIGH[0]))
    axis.set_ylim(float(WORKSPACE_LOW[1]), float(WORKSPACE_HIGH[1]))
    axis.set_zlim(float(-WORKSPACE_HIGH[2]), float(-WORKSPACE_LOW[2]))
    axis.invert_zaxis()
    axis.set_box_aspect((4.0, 4.0, 3.0))
    axis.view_init(elev=23, azim=-58)
    axis.set_xlabel("x (m)", labelpad=2, fontsize=8)
    axis.set_ylabel("y (m)", labelpad=2, fontsize=8)
    axis.set_zlabel("Depth (m)", labelpad=3, fontsize=8)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.zaxis.set_major_locator(MaxNLocator(5))
    axis.grid(True)
    for axis_obj in (axis.xaxis, axis.yaxis, axis.zaxis):
        axis_obj._axinfo["grid"].update(
            {
                "color": (0.72, 0.76, 0.79, 0.22),
                "linewidth": 0.35,
                "linestyle": ":",
            }
        )
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor("#FAFCFD")
        pane.set_edgecolor("#D8E0E6")
        pane.set_alpha(0.14)

    _draw_workspace_box(axis)

    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        samples = method_paths.get(method)
        if not samples:
            continue
        best_path, _ = min(samples, key=lambda item: item[1])
        plotted = _depth_path(best_path)
        color = METHOD_COLORS.get(method, "#444444")
        linewidth = 2.5 if method == "LLMOperatorDE" else 1.45
        alpha = 1.0 if method == "LLMOperatorDE" else 0.86
        axis.plot(
            plotted[:, 0],
            plotted[:, 1],
            plotted[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=linewidth,
                label=METHOD_LABELS.get(method, method),
            )
        )

    _draw_obstacles(axis, scenario_data["obstacles"])
    _draw_current(axis, scenario_data)

    start = _depth_path(np.asarray(scenario_data["start"], dtype=float).reshape(1, 3))[0]
    goal = _depth_path(np.asarray(scenario_data["goal"], dtype=float).reshape(1, 3))[0]
    axis.scatter(*start, color="#2CA02C", s=38, marker="o", edgecolor="white", linewidth=0.6, depthshade=False)
    axis.scatter(*goal, color="#222222", s=48, marker="*", edgecolor="white", linewidth=0.6, depthshade=False)

    handles.extend(
        [
            Line2D([0], [0], color="#56C5D8", linewidth=1.1, marker=">", markersize=4, label="Current direction"),
            Patch(facecolor="#D96B6B", edgecolor="#B84A4A", alpha=0.28, label="Obstacle"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2CA02C", markeredgecolor="white", markersize=7, label="Start"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#222222", markeredgecolor="white", markersize=9, label="Goal"),
        ]
    )
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.98),
        borderaxespad=0.0,
        frameon=False,
        fontsize=6.2,
        handlelength=1.6,
        labelspacing=0.32,
        handletextpad=0.55,
        borderpad=0.15,
        columnspacing=0.7,
        ncols=2,
    )
    axis.set_title(
        SCENARIO_TITLES.get(scenario, scenario),
        fontsize=9.5,
        fontweight="bold",
        color="#20364A",
        pad=3,
    )
    fig.subplots_adjust(left=0.01, right=0.82, bottom=0.035, top=0.89)

    # Add a restrained frame around the 3-D plotting area. This keeps the
    # publication-style light background while making the panel boundary clear
    # in PDF/SVG exports and when the figure is placed on a white page.
    plot_box = axis.get_position()
    fig.add_artist(
        Rectangle(
            (plot_box.x0, plot_box.y0),
            plot_box.width,
            plot_box.height,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#AEB8C1",
            linewidth=0.75,
            zorder=1000,
            clip_on=False,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    import sys

    project = Path(__file__).resolve().parents[2]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from experiments.underwater_robot.experiment_config import CONFIG

    root = Path(CONFIG.output_root) / "underwater_comparison"
    for scenario in CONFIG.scenarios:
        scenario_root = root / scenario
        if scenario_root.exists():
            plot_3d_paths(scenario_root, scenario, scenario_root / "underwater_paths_3d.png")


__all__ = ["plot_3d_paths"]


if __name__ == "__main__":
    main()
