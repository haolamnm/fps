import argparse
import logging
import sys

from ..services.common.utils import get_logger
from .constants import FAILURE_EXIT_CODE, PORT_MAPPING, SERVICE_MAPPING, SUCCESS_EXIT_CODE

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a service in server mode")
    parser.add_argument(
        "service",
        type=str,
        help="service to start (e.g., features-clip, features-clip2video)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host to bind the service to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="port to bind the service to (default: will use port from mapping if not provided)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable verbose logging (default: False)",
    )

    # Parse known args first
    args, unknown = parser.parse_known_args()

    # Add service-specific arguments back to the parser
    # These will be passed through to the service
    service_parser = argparse.ArgumentParser(add_help=False)

    # Common service-specific arguments
    if args.service == "features-clip" or args.service == "clip-openai":
        service_parser.add_argument(
            "--model-name",
            type=str,
            default="openai/clip-vit-large-patch14",
            choices=[
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-large-patch14",
                "openai/clip-vit-large-patch14-336",
            ],
            help="name of the CLIP model to use (default: openai/clip-vit-large-patch14)",
        )
    elif args.service == "features-openclip" or args.service in ["clip-laion", "clip-datacomp"]:
        service_parser.add_argument(
            "--model-name",
            default="ViT-L-14",
            type=str,
            choices=["ViT-L-14", "ViT-B-32", "ViT-B-16"],
            help="model name to use for feature extraction (default: ViT-L-14)",
        )
        service_parser.add_argument(
            "--pretrained",
            default="laion2b_s32b_b82k",
            type=str,
            choices=[
                "laion2b_s32b_b82k",
                "datacomp_xl_s13b_b90k",
            ],
            help="pretrained model to use for feature extraction (default: laion2b_s32b_b82k)",
        )
    elif args.service == "features-dinov2" or args.service == "dinov2":
        service_parser.add_argument(
            "--model-name",
            default="dinov2_vitl14",
            type=str,
            choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
            help="name of the DINOv2 model to use (default: dinov2_vitl14)",
        )

    # Parse remaining args with the service-specific parser
    service_args, remaining = service_parser.parse_known_args(unknown)

    # Combine the args
    for key, value in vars(service_args).items():
        if value is not None:
            setattr(args, key, value)

    # Handle any remaining unknown args
    if remaining:
        logger.warning(f"Unrecognized arguments: {remaining}")

    return args


def get_service_module(service_name: str) -> str:
    """
    Convert a service name to its module path.
    For example, 'features-clip' or 'clip-openai' -> 'fps.services.analysis.features-clip.serve'
    """
    # Handle aliases first
    if service_name in SERVICE_MAPPING:
        service_name = f"features-{SERVICE_MAPPING[service_name]}"

    # Handle special case for index services
    if service_name == "faiss-index-manager" or service_name == "faiss":
        return "fps.services.index.faiss-index-manager.serve"

    if not service_name.startswith("features-"):
        service_name = f"features-{service_name}"

    return f"fps.services.analysis.{service_name}.serve"


def get_service_args(args: argparse.Namespace) -> list[str]:
    """
    Convert the args to a list of strings for the service command.
    """
    cmd_args: list[str] = []

    # Add host and port
    cmd_args.extend(["--host", args.host])

    # Use the port from mapping if not explicitly provided
    if args.port is None:
        service_name = args.service
        if service_name in SERVICE_MAPPING:
            service_name = f"features-{SERVICE_MAPPING[service_name]}"

        if service_name in PORT_MAPPING:
            cmd_args.extend(["--port", str(PORT_MAPPING[service_name])])
        else:
            # Default port if not in mapping
            cmd_args.extend(["--port", "8080"])
    else:
        cmd_args.extend(["--port", str(args.port)])

    # Add service-specific args
    for key, value in vars(args).items():
        # Skip non-service specific args
        if key in ["service", "host", "port", "verbose"]:
            continue

        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd_args.append(f"--{key}")
            else:
                cmd_args.extend([f"--{key.replace('_', '-')}", str(value)])

    return cmd_args


def main() -> int:
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Get the module name for the service
    service_module = get_service_module(args.service)

    # Get the command line arguments for the service
    service_args = get_service_args(args)

    # Log the service being started
    logger.info(f"Starting service: {service_module}")
    if args.verbose:
        logger.debug(f"Command arguments: {' '.join(service_args)}")

    try:
        # Set sys.argv for the module
        sys.argv = [service_module] + service_args

        # Import and run the module
        module_name = service_module.split(".")
        if module_name[-1] == "serve":
            service = __import__(service_module, fromlist=["*"])
            if hasattr(service, "main"):
                return service.main()
            return SUCCESS_EXIT_CODE
        else:
            logger.error(f"Invalid service module: {service_module}")
            return FAILURE_EXIT_CODE
    except ImportError as e:
        logger.error(f"Failed to import service {service_module}: {str(e)}")
        return FAILURE_EXIT_CODE
    except Exception as e:
        logger.error(f"Error starting service {service_module}: {str(e)}")
        return FAILURE_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
