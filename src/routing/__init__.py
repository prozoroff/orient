"""Поиск и визуализация маршрутов по сегментированной карте ориентирования."""

from src.routing.api import (
    RoutingResult,
    compute_routes,
    find_routes_on_map,
    prepare_segmentation,
)
from src.routing.config import (
    CLASS_LABELS_RU,
    CLASS_NAMES,
    DEFAULT_CONTOUR_PENALTY,
    DEFAULT_SPEEDS_MPS,
    copy_speeds,
)
from src.routing.cost_model import CostGrid, build_cost_grid
from src.routing.geo import MapMeta, load_meta, meters_per_pixel_from_scale
from src.routing.inference import load_rgb, segment_map
from src.routing.pathfinding import Route, SearchTrace, astar_traced, find_k_routes
from src.routing.segments import (
    describe_segments,
    segments_to_dataframe,
    summarize_route,
)
from src.routing.visualize import (
    animate_path_search,
    animate_segmentation,
    format_route_description,
    render_search_frames,
    render_segmentation_map,
    render_speed_map,
    show_route_table,
    show_routes,
    show_segmentation,
    show_speed_map,
    speed_legend_handles,
)

__all__ = [
    "CLASS_LABELS_RU",
    "CLASS_NAMES",
    "DEFAULT_CONTOUR_PENALTY",
    "DEFAULT_SPEEDS_MPS",
    "CostGrid",
    "MapMeta",
    "Route",
    "RoutingResult",
    "SearchTrace",
    "animate_path_search",
    "animate_segmentation",
    "astar_traced",
    "build_cost_grid",
    "compute_routes",
    "copy_speeds",
    "describe_segments",
    "find_k_routes",
    "find_routes_on_map",
    "format_route_description",
    "load_meta",
    "load_rgb",
    "meters_per_pixel_from_scale",
    "prepare_segmentation",
    "render_search_frames",
    "render_segmentation_map",
    "render_speed_map",
    "segment_map",
    "segments_to_dataframe",
    "show_route_table",
    "show_routes",
    "show_segmentation",
    "show_speed_map",
    "speed_legend_handles",
    "summarize_route",
]
