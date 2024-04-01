"""This module is in prototyping stage and is not used in the main program."""
import logging


# Set up console logging for the package
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def reconstruct_edges_geometry():
    pass


def assign_geometry_to_edges():
    pass