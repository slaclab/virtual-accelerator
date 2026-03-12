import argparse
from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model, get_cu_hxr_cheetah_model
from lume_pva.runner import Runner


def main():
    parser = argparse.ArgumentParser(
        description="Run the CU HXR model with BMAD or CHEETAH backend"
    )
    parser.add_argument(
        "model",
        choices=["cu_hxr_bmad", "cu_hxr_cheetah"],
        help="Model backend to run (cu_hxr_bmad or cu_hxr_cheetah)",
    )
    
    args = parser.parse_args()
    
    # Get the appropriate model based on user input
    if args.model == "cu_hxr_bmad":
        model = get_cu_hxr_bmad_model()
    elif args.model == "cu_hxr_cheetah":  # cu_hxr_cheetah
        model = get_cu_hxr_cheetah_model()
    else:
        raise ValueError("Invalid model choice. Please choose 'cu_hxr_bmad' or 'cu_hxr_cheetah'.")
    
    # Run the model
    runner = Runner(model)
    runner.run()


if __name__ == "__main__":
    main()