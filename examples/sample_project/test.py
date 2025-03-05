import ruamel.yaml as yaml
import argparse
import pathlib
import sys
from dreamkit import Agent 

sys.path.append(str(pathlib.Path(__file__).parent))

def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+")
args, remaining = parser.parse_known_args()
configs = yaml.safe_load(
    (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
defaults = {}
for name in name_list:
    recursive_update(defaults, configs[name])
parser = argparse.ArgumentParser()
for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = args_type(value)
    parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

# Initialize the Agent
args = parser.parse_args(remaining)
if args.logdir is None:
    args.logdir = str(pathlib.Path().resolve())
agent = Agent(args)

# sh xvfb_run.sh python test.py --configs dmc_vision --task dmc_walker_walk --logdir ./log