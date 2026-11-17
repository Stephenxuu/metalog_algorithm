"""Feasibility test, moments, and best feasible fit for the metalog 2.0 distribution.

Companion code to the paper "The Metalog Distribution: Properties and Feasible Estimation"
by Manel Baucells, Lonnie Chrisman, Thomas W. Keelin, and Stephen Xu.

Copyright (c) 2025 Stephen Xu. Released under the MIT License; see the LICENSE file for the full text.
"""

from .algo_astar import calculate_Y, find_a_star
from .feasibility_stats import central_moment, feasible, raw_moment, summary_stats

__version__ = "0.1.0"

__all__ = [
    "find_a_star",
    "calculate_Y",
    "feasible",
    "summary_stats",
    "raw_moment",
    "central_moment",
]
