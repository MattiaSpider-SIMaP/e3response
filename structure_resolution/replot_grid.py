#!/usr/bin/env python3
"""Regenerate the grid-search figures from a saved bundle — no model, no GPU.

A grid-search run writes ``plot_data.npz`` + ``plot_meta.json`` into its results directory
(via `e3response.sr_viz.save_bundle`). This tool reloads that bundle and redraws the result
figure and the descent-path GIF, so you can iterate on the look — colours, framework opacity,
GIF frame count / fps — in seconds without re-running the optimiser. Tweak the drawing code in
``e3response/sr_viz.py``, then re-run this.

Usage:
    python structure_resolution/replot_grid.py <results_dir> [options]

    <results_dir>   directory holding plot_data.npz + plot_meta.json
    --figure/--no-figure   redraw the result PNG              (default: on)
    --gif/--no-gif         redraw the descent GIF             (default: on)
    --out-dir DIR          where to write (default: <results_dir>)
    --dpi N                figure DPI                         (default: 150)
    --fps N / --top-n N / --max-frames N   GIF controls (default: bundle values)
    --cand-cmap NAME       candidate-loss colormap            (default: viridis_r)
    --path-cmap NAME       descent-path colormap              (default: plasma)
    --framework-alpha F    framework opacity in the figure    (default: 0.18)
"""
import argparse
import os
import sys

from e3response import sr_viz


def main(argv=None):
    p = argparse.ArgumentParser(description="Regenerate grid-search figures from a saved bundle.")
    p.add_argument("results_dir", help="directory with plot_data.npz + plot_meta.json")
    p.add_argument("--figure", dest="figure", action="store_true", default=True)
    p.add_argument("--no-figure", dest="figure", action="store_false")
    p.add_argument("--gif", dest="gif", action="store_true", default=True)
    p.add_argument("--no-gif", dest="gif", action="store_false")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--cand-cmap", default="viridis_r")
    p.add_argument("--path-cmap", default="plasma")
    p.add_argument("--framework-alpha", type=float, default=0.18)
    args = p.parse_args(argv)

    if not os.path.isfile(os.path.join(args.results_dir, sr_viz.DATA_NAME)):
        sys.exit(f"No {sr_viz.DATA_NAME} in {args.results_dir!r} — is it a grid-search results dir?")

    b = sr_viz.load_bundle(args.results_dir)
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.figure:
        fig_path = os.path.join(out_dir, "structure_resolution_result.png")
        sr_viz.grid_result_figure(b, fig_path, cand_cmap=args.cand_cmap,
                                  framework_alpha=args.framework_alpha, dpi=args.dpi)
        print(f"  figure → {fig_path}")

    if args.gif:
        gif_path = os.path.join(out_dir, "structure_resolution.gif")
        sr_viz.grid_descent_gif(
            b, gif_path,
            top_n=args.top_n if args.top_n is not None else int(b.get("gif_top_n", 8)),
            max_frames=args.max_frames if args.max_frames is not None else int(b.get("gif_max_frames", 120)),
            fps=args.fps if args.fps is not None else int(b.get("gif_fps", 15)),
            path_cmap=args.path_cmap)
        print(f"  gif → {gif_path}")


if __name__ == "__main__":
    main()
