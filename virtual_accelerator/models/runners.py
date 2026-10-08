import argparse
import logging

from virtual_accelerator.registry import get_model, list_models
from virtual_accelerator.utils.optional_dependencies import import_optional_symbol


def _build_model(args):
    kwargs = {}
    if args.n_particles is not None:
        kwargs["n_particles"] = args.n_particles
    if args.track_beam:
        kwargs["track_beam"] = True

    return get_model(
        args.model,
        start_ele=args.start_ele,
        end_ele=args.end_ele,
        handoff_loc=args.handoff_loc,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a virtual accelerator model. MODEL is any registry name or "
            "chain alias (see --list-models)."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Registry name (e.g. bmad_cu_hxr) or chain alias (e.g. fast_cu_hxr_s2e).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the registry table and exit.",
    )
    parser.add_argument(
        "--start-ele",
        default=None,
        help="Start element. Default: model's standard start.",
    )
    parser.add_argument(
        "--end-ele",
        default=None,
        help="End element. Default: model's standard end.",
    )
    parser.add_argument(
        "--handoff-loc",
        default=None,
        help="Element handing the beam between stages of a chain. Default: inferred.",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="Particle count for models that sample one.",
    )
    parser.add_argument(
        "--track-beam",
        action="store_true",
        help="Force beam tracking on for a single Bmad model (chains always track).",
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

    if args.list_models:
        print(list_models())
        return

    if args.model is None:
        parser.error("the following arguments are required: model (or --list-models)")

    Runner = import_optional_symbol(
        "lume_pva.runner",
        "Runner",
        feature="virtual accelerator runner CLI",
        extra="pva",
    )

    model = _build_model(args)
    Runner(model).run()


if __name__ == "__main__":
    main()
