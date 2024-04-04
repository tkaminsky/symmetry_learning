from environment.symmetry_move import SymmetryMoveEnv
from objects.nested_shape import NestedShape
from objects.shapes.polygon import RegularPolygon
import numpy as np
import pygame
import matplotlib.pyplot as plt
from helpers.blueprints import generate_valid_regular_polygon_blueprints, triangle_bp, L_bp
import argparse
from controllers.manual_controller import manualController
from controllers.oracle_controller import oracleController
from algos.bc import bcRunner

# Trains a BC model to do symmetry learning
def bc_control(config):
    # bp1, bp2 = generate_valid_regular_polygon_blueprints()
    bp1, bp2 = L_bp
    env = SymmetryMoveEnv(bp1, bp2, render_mode="human")
    bcRunner(env)

# Manually control the agent with arrow keys
def manual_control(config):
    # bp1, bp2 = generate_valid_regular_polygon_blueprints()
    bp1, bp2 = L_bp
    env = SymmetryMoveEnv(bp1, bp2)
    manualController(env)

# Perfect oracle control (ignores object symmetry)
def oracle_control(config):
    bp1, bp2 = generate_valid_regular_polygon_blueprints()
    env = SymmetryMoveEnv(bp1, bp2)
    oracleController(env)


def main(config):
    if config["control"] == 'manual':
        manual_control(config)
    elif config["control"] == 'oracle':
        oracle_control(config)
    elif config["control"] == 'bc':
        bc_control(config)
    else:
        raise ValueError("Invalid control: " + config["control"])
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Specify the control type
    parser.add_argument(
        "-c", "--control", type=str, default="manual", help="Control type"
    )
    
    args = parser.parse_args()
    # Turn args into a dictionary
    config = vars(args)
    main(config)