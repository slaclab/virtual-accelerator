"""
Script to generate the slac_variable_config.yaml file programmatically.
"""

import yaml
from pathlib import Path


def scalar_var(unit=None, read_only=True, default_value=None):
    """Create a ScalarVariable configuration."""
    config = {
        "variable_class": "ScalarVariable",
        "read_only": read_only,
    }
    if unit:
        config["unit"] = unit
    if default_value is not None:
        config["default_value"] = default_value
    return config


def enum_var(options, default_value="Ready"):
    """Create an EnumVariable configuration."""
    return {
        "variable_class": "EnumVariable",
        "read_only": True,
        "default_value": default_value,
        "options": options,
    }


def create_magnet_config(b_unit):
    """Create configuration for magnets (Quadrupole, Solenoid, Correctors)."""
    ctrl_options = {
        0: "Ready",
        1: "TRIM",
        2: "PERTURB",
        3: "BCON_TO_BDES",
        4: "BACT_TO_BDES",
    }
    return {
        "BCTRL": scalar_var(unit=b_unit, read_only=False),
        "BCTRL.DRVL": scalar_var(unit=b_unit, read_only=True),
        "BCTRL.DRVH": scalar_var(unit=b_unit, read_only=True),
        "BACT": scalar_var(unit=b_unit, read_only=True),
        "BDES": scalar_var(unit=b_unit),
        "BMIN": scalar_var(unit=b_unit, read_only=True),
        "BMAX": scalar_var(unit=b_unit, read_only=True),
        "STATCTRLSUB.T": scalar_var(read_only=True),
        "CTRL": enum_var(ctrl_options),
    }


def generate_slac_variable_config():
    """Generate the SLAC variable configuration dictionary."""
    
    config = {
        "BPM": {
            "X": scalar_var(unit="mm", default_value=None),
            "Y": scalar_var(unit="mm"),
        },
        "Quadrupole": create_magnet_config("kG"),
        "Solenoid": create_magnet_config("kG-m"),
        "HorizontalCorrector": create_magnet_config("kG-m"),
        "VerticalCorrector": create_magnet_config("kG-m"),
        "Screen": {
            "Image:ArrayData": {
                "unit": "pixel",
                "read_only": True,
                "variable_class": "NDVariable",
                "shape": None,
            },
            "PNEUMATIC": scalar_var(unit="bool", read_only=False),
            "Image:ArraySize1_RBV": scalar_var(unit="pixel", read_only=True),
            "Image:ArraySize0_RBV": scalar_var(unit="pixel", read_only=True),
            "RESOLUTION": scalar_var(unit="pixel/mm", read_only=True),
        },
        "Lcavity_Overlay": {
            "ENLD": scalar_var(unit="MeV", read_only=False),
            "PHAS": scalar_var(unit="Deg_S", read_only=False),
            "BEAMCODE1_STAT": scalar_var(unit="Boolean", read_only=False),
        },
        "TransverseDeflectingCavity": {
            "AREQ": scalar_var(unit="degrees", read_only=False),
            "PREQ": scalar_var(unit="MV", read_only=False),
        },
    }
    
    return config


def save_config_to_yaml(config, output_path):
    """Save the configuration dictionary to a YAML file."""
    
    class NullPresenter(yaml.SafeDumper):
        """Custom YAML dumper to represent None as NULL."""
        pass
    
    def represent_none(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', 'NULL')
    
    NullPresenter.add_representer(type(None), represent_none)
    
    with open(output_path, 'w') as f:
        yaml.dump(
            config,
            f,
            Dumper=NullPresenter,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    
    print(f"Configuration saved to {output_path}")


if __name__ == "__main__":
    # Generate the configuration
    config = generate_slac_variable_config()
    
    # Save to the default location
    script_dir = Path(__file__).parent
    output_path = script_dir / "slac_variable_config.yaml"
    
    save_config_to_yaml(config, output_path)
