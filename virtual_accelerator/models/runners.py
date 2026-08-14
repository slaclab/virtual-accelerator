import argparse

from virtual_accelerator.utils.optional_dependencies import import_optional_symbol

import logging


def main():
    parser = argparse.ArgumentParser(description="Run a virtual accelerator model")
    choices = [
        "cu_hxr_bmad",
        "cu_hxr_staged",
        "facet_bmad",
        "facet_staged",
        "cu_hxr_zfel",
    ]
    parser.add_argument(
        "model",
        choices=choices,
        help="Model backend to run (cu_hxr_bmad, cu_hxr_staged, facet_bmad, facet_staged, or cu_hxr_zfel)",
    )
    parser.add_argument(
        "--end-element",
        default="END",
        help="End lattice element for BMAD models (default: END)",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=10000,
        help="Number of particles for model if used (default: 10000)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.getLogger("pytao").setLevel(logging.WARNING)

    Runner = import_optional_symbol(
        "lume_pva.runner",
        "Runner",
        feature="virtual accelerator runner CLI",
        extra="pva",
    )

    # Get the appropriate model based on user input
    if args.model == "cu_hxr_bmad":
        from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model

        model = get_cu_hxr_bmad_model(end_element=args.end_element, track_beam=True)
    elif args.model == "cu_hxr_staged":
        from virtual_accelerator.models.cu_hxr import get_cu_hxr_staged_model

        model = get_cu_hxr_staged_model(
            end_element=args.end_element, n_particles=args.n_particles
        )
    elif args.model == "facet_bmad":
        from virtual_accelerator.models.facet2 import get_facet_bmad_model

        model = get_facet_bmad_model(end_element=args.end_element, track_beam=True)
    elif args.model == "facet_staged":
        from virtual_accelerator.models.facet2 import get_facet_staged_model

        model = get_facet_staged_model(
            end_element=args.end_element, n_particles=args.n_particles
        )
    elif args.model == "cu_hxr_zfel":
        from virtual_accelerator.models.cu_hxr_zfel import (
            get_cu_hxr_zfel_runner,
        )

        runner = get_cu_hxr_zfel_runner(Runner)
    else:
        raise ValueError(f"Invalid model choice. Please choose one of {choices}.")

    # Run the model
    if args.model != "cu_hxr_zfel":
        runner = Runner(model)
    runner.run()


if __name__ == "__main__":
    main()
