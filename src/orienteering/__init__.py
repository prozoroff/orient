"""Планирование score-O дистанции: макс. баллов за отведённое время."""

from src.orienteering.api import build_legs, plan_course, plan_course_from_parts
from src.orienteering.scoring import points_from_number
from src.orienteering.solver import EXACT_DP_MAX_N, DpTrace, SolverResult, solve_orienteering
from src.orienteering.types import (
    CoursePlan,
    ScoredControl,
    controls_table_rows,
    scale_speeds_to_open_land,
)
from src.orienteering.visualize import (
    animate_course_pipeline,
    draw_course_plan,
    print_plan_summary,
    render_course_legs_frames,
    render_course_pipeline_frames,
    render_edge_matrix_frames,
    render_final_tour_frames,
    render_solver_frames,
    render_speed_map,
    show_course_plan,
    show_course_with_speed,
)

__all__ = [
    "EXACT_DP_MAX_N",
    "CoursePlan",
    "DpTrace",
    "ScoredControl",
    "SolverResult",
    "animate_course_pipeline",
    "build_legs",
    "controls_table_rows",
    "draw_course_plan",
    "plan_course",
    "plan_course_from_parts",
    "points_from_number",
    "print_plan_summary",
    "render_course_legs_frames",
    "render_course_pipeline_frames",
    "render_edge_matrix_frames",
    "render_final_tour_frames",
    "render_solver_frames",
    "render_speed_map",
    "scale_speeds_to_open_land",
    "show_course_plan",
    "show_course_with_speed",
    "solve_orienteering",
]
