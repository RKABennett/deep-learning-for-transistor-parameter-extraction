"""
Command Line Interface for transistor parameter extraction.
"""

import argparse
import sys
import os
import yaml
from pathlib import Path


def load_config(config_path="configs/default_config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)


def train():
    """Train neural network models"""
    parser = argparse.ArgumentParser(
        description="Train transistor parameter extraction models"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--experiment", "-e", default="2d_transistor", help="Experiment name"
    )

    args = parser.parse_args()

    print(f"Starting training with config: {args.config}")
    print(f"Experiment: {args.experiment}")

    config = load_config(args.config)

    # Import and run training
    sys.path.append(f"experiments/{args.experiment}")
    try:
        import train_model

        print("Training completed successfully!")
    except ImportError as e:
        print(f"Error importing training module: {e}")
        sys.exit(1)


def test():
    """Test trained models"""
    parser = argparse.ArgumentParser(
        description="Test transistor parameter extraction models"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--experiment", "-e", default="2d_transistor", help="Experiment name"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        choices=["forward", "inverse"],
        default="forward",
        help="Model type to test",
    )

    args = parser.parse_args()

    print(f"Starting testing with config: {args.config}")
    print(f"Experiment: {args.experiment}")
    print(f"Model type: {args.model_type}")

    config = load_config(args.config)

    # Import and run testing
    sys.path.append(f"experiments/{args.experiment}")
    try:
        if args.model_type == "forward":
            import test_forward_0_save_fits
            import test_forward_1_plot_fits
        else:
            import test_inverse_0_save_fits
            import test_inverse_1_param_extract
            import test_inverse_2_plot_fits
        print("Testing completed successfully!")
    except ImportError as e:
        print(f"Error importing test module: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: python cli.py [train|test] [options]")
        sys.exit(1)
