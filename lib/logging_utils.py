#!/usr/bin/env python3
"""Custom logging utilities for Red Bull Editions Generator.

Adds a VERBOSE logging level between INFO and DEBUG for detailed operational logging.
Provides standardized logging setup functions for consistent output across modules.
"""
import logging

# Define VERBOSE level (between INFO=20 and DEBUG=10)
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")


def setup_basic_logging() -> None:
    """Setup basic logging configuration for console output.

    Configures logging with INFO level and simple message format.
    This should be called once at the start of the application.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def verbose(self, message, *args, **kwargs):
    """Log a message with severity 'VERBOSE'.

    Args:
        message: Log message format string
        *args: Format arguments
        **kwargs: Additional keyword arguments for logging
    """
    if self.isEnabledFor(VERBOSE):
        logging.log(VERBOSE, message, *args, **kwargs)


# Add verbose method to Logger class
logging.Logger.verbose = verbose


def setup_logger(name: str, enable_verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Setup a logger with appropriate level.

    Creates a logger with the correct level based on verbosity settings.
    Hierarchy: DEBUG (10) > VERBOSE (15) > INFO (20)

    Args:
        name: Logger name (typically class name)
        enable_verbose: Enable VERBOSE level logging
        debug: Enable DEBUG level logging (includes VERBOSE)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if debug:
        logger.setLevel(logging.DEBUG)
    elif enable_verbose:
        logger.setLevel(VERBOSE)
    else:
        logger.setLevel(logging.INFO)

    return logger
