# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Shared logging configuration for utility scripts running in cluster jobs.

This module provides standardized logging configuration for utility scripts that
run inside Slurm jobs. Use this for any script that performs data processing on
the cluster (as opposed to orchestration code which uses console.py).

Example:
    >>> from nvflow.utils.logging_setup import setup_logger
    >>>
    >>> logger = setup_logger(__name__)
    >>> logger.info("Processing started")
    >>> logger.error("Failed to process record")
"""

import logging
import sys


def setup_logger(
    name: str = __name__,
    level: str = "INFO",
    format_string: str | None = None,
) -> logging.Logger:
    """Configure logging for cluster utility scripts.

    Creates a logger with stdout handler configured for Slurm job output.
    All log messages go to stdout (captured by Slurm) with timestamps and
    level indicators.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Log level - DEBUG, INFO, WARNING, ERROR, CRITICAL
        format_string: Custom format string (uses default if None)

    Returns:
        Configured logger instance ready to use

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing 1000 records")
        2025-11-23 10:15:32 | INFO     | Processing 1000 records

        >>> logger = setup_logger(__name__, level="DEBUG")
        >>> logger.debug("Detailed diagnostic info")
        2025-11-23 10:15:33 | DEBUG    | Detailed diagnostic info
    """
    if format_string is None:
        # Default format: timestamp | level (8 chars) | message
        format_string = "%(asctime)s | %(levelname)-8s | %(message)s"

    # Create or get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler - outputs to stdout (captured by Slurm)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Formatter with timestamps
    formatter = logging.Formatter(fmt=format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    return logger
