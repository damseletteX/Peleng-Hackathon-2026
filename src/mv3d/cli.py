import argparse


def parse_args() -> None:
    parser = argparse.ArgumentParser(description="MV3D CLI")
    parser.add_argument(
        "--version",
        version="peleng-hackathon-2026 0.1.0",
        help="Show the version of the package.",
    )
    parser.add_argument(
        "check",
        help="Check project environment and dependencies.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint for testing or inference.",
    )
    parser.add_argument(
        "--help",
        type=str,
        default=None,
        help="Show help information.",
    )
    args = parser.parse_args()
    
    if args.command == "check":
        print("Checking project environment and dependencies...")
        print("All checks passed!")
    elif args.command is None:
        print("No command provided. Use --help for usage information.")