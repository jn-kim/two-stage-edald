import os
import json
import yaml
from argparse import ArgumentParser, Namespace
from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from utils.utils import setup_seed

class Arguments:
    def __init__(self):
        parser = ArgumentParser("two-stage-edald")
        self.default_model_flags = model_and_diffusion_defaults()
        add_dict_to_argparser(parser, self.default_model_flags)
        
        # Specify path here for configuration files (yaml files)
        parser.add_argument("--dataset_path", type=str, default="datasets/")
        parser.add_argument("--pt_path", type=str, default="checkpoints/")
        # Experiment directory
        parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
        parser.add_argument("--use_wandb", type=Arguments.str2bool, default=True)
        parser.add_argument("--resume", type=Arguments.str2bool, default=True)
        
        # Dataset & Training Settings
        parser.add_argument("--dataset_name", type=str, choices=["ade", "camvid", "cityscapes", "pascal"], required=True)
        parser.add_argument("--data_usage", type=float, default=100)
        parser.add_argument("--network_name", type=str, default="ddpm", choices=["ddpm"])
        parser.add_argument("--pretrained_model", type=str, default="imagenet_256", choices=["imagenet_256", "lsun_bedroom"])
        parser.add_argument("--seed", type=int, default=0)

        # Active Learning
        parser.add_argument("--keep_global", type=float, default=0.0)
        parser.add_argument("--uncertainty", type=str, default="entropy_dald",
                            choices=["entropy_bald", "bald", "power_bald", "balentacq", "dald", "power_dald", "entropy_dald", "entropy", "margin_sampling"])
        
        # Bayesian approximation method
        parser.add_argument("--use_mc_dropout", type=Arguments.str2bool, default=False)
        parser.add_argument("--use_diffusion_stochastic", type=Arguments.str2bool, default=False)
        parser.add_argument("--single_forward", type=Arguments.str2bool, default=False)

        # Additional Active Learning Settings
        parser.add_argument("--n_stages", type=int, default=10)
        parser.add_argument("--nth_query", type=int, default=1)

        # MaxHerding settings
        parser.add_argument("--local_candidate", type=int, default=10000)
        parser.add_argument("--local_budget", type=int, default=50)
        parser.add_argument("--local_budget_init", type=int, default=50, help="Local budget for 0 round")
        parser.add_argument("--final_budget_factor", type=float, default=0.1)
        parser.add_argument("--normalize_features", type=Arguments.str2bool, default=True)

        # MC Dropout Settings
        parser.add_argument("--mc_dropout_p", type=float, default=0.2)
        parser.add_argument("--mc_n_steps", type=int, default=5)
        
        # Noisy-x (random seed) driven DALD Settings
        parser.add_argument("--num_noisy_x", type=int, default=5)
        
        # DDPM Settings
        parser.add_argument("--steps", type=str, default="50,150,250")
        parser.add_argument("--blocks", type=str, default="5,8,12,17")
        
        # training config
        parser.add_argument("--n_epochs", type=int, default=50)
        parser.add_argument("--train_mode", type=str, default="early_stop", choices=["early_stop"])
        self.parser = parser
    
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v_lower = v.lower()
        if v_lower in ("true", "t", "y", "1"):
            return True
        elif v_lower in ("false", "f", "n", "0"):
            return False
    
    @staticmethod
    def resolve_env_vars(obj, env):
        import string
        if isinstance(obj, dict):
            return {k: Arguments.resolve_env_vars(v, env) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Arguments.resolve_env_vars(i, env) for i in obj]
        elif isinstance(obj, str):
            return string.Template(obj).safe_substitute(env)
        else:
            return obj
    
    def parse_args(self, verbose: bool = False):
        args = self.parser.parse_args()
        setup_seed(args.seed)

        env_dict = {
            "DATASET_PATH": args.dataset_path,
            "CHECKPOINT_DIR": args.checkpoint_dir,
            "PT_PATH": args.pt_path
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_yaml = os.path.join(script_dir, "configs/dataset_config.yaml")
        training_yaml = os.path.join(script_dir, "configs/training_config.yaml")
        ddpm_yaml = os.path.join(script_dir, "configs/ddpm.yaml")
        uncertainty_yaml = os.path.join(script_dir, "configs/uncertainty_config.yaml")

        with open(dataset_yaml, 'r') as f:
            raw_dataset_config = yaml.safe_load(f)["datasets"]
            dataset_config = self.resolve_env_vars(raw_dataset_config, env_dict)

        with open(training_yaml, 'r') as f:
            raw_training_config = yaml.safe_load(f)["training"]
            training_config = self.resolve_env_vars(raw_training_config, env_dict)

        with open(ddpm_yaml, 'r') as f:
            raw_ddpm_config = yaml.safe_load(f)["ddpm"]
            ddpm_config = self.resolve_env_vars(raw_ddpm_config, env_dict)

        with open(uncertainty_yaml, 'r') as f:
            uncertainty_config = yaml.safe_load(f)["uncertainty_configs"]

        dataset_key = args.dataset_name
        args_dict = vars(args)

        # Auto-configure flags based on uncertainty method
        uncertainty_method = args_dict.get("uncertainty")
        
        if uncertainty_method in uncertainty_config.get("bald_methods", []):
            args_dict["use_mc_dropout"] = True
        
        if uncertainty_method in uncertainty_config.get("dald_methods", []):
            args_dict["use_diffusion_stochastic"] = True
        
        if uncertainty_method in uncertainty_config.get("single_forward_methods", []):
            args_dict["single_forward"] = True

        dataset_settings = dataset_config.get(dataset_key, {})
        training_settings = training_config.get(dataset_key, {})

        for key, value in {**dataset_settings, **training_settings}.items():
            args_dict.setdefault(key, value)
        if args_dict.get("n_epochs") is None and "n_epochs" in training_settings:
            args_dict["n_epochs"] = training_settings["n_epochs"]

        # ddpm configuration
        segmentation_head_config = ddpm_config.get("segmentation_head", {})
        args_dict["segmentation_head"] = segmentation_head_config.get("type")
        args_dict["model_num"] = segmentation_head_config.get("model_num")

        if args.network_name == "ddpm":
            ddpm_model = args.pretrained_model
            assert ddpm_model in ddpm_config["models"], f"Pretrained model '{ddpm_model}' not found in ddpm.yaml"
            model_config = ddpm_config["models"][ddpm_model]
            args_dict["model_path"] = model_config["model_path"]

            model_flags = self.default_model_flags.copy()
            model_flags.update(model_config["model_flags"])
            args_dict.update(model_flags)
            args_dict.pop("model_flags", None)

            if args.steps is not None:
                args_dict["steps"] = list(map(int, args.steps.split(',')))
            if args.blocks is not None:
                args_dict["blocks"] = list(map(int, args.blocks.split(',')))

            for key, value in ddpm_config.items():
                if key != "models":
                    args_dict.setdefault(key, value)

            args_dict["decoder_features"] = ddpm_config.get("decoder_features", {})
            selected_blocks = args_dict.get("blocks", [])
            args_dict["feature_channels"] = [args_dict["decoder_features"][str(b)][0] for b in selected_blocks]
            dim = sum(args_dict["decoder_features"][str(b)][0] for b in selected_blocks) * len(args_dict["steps"])
            args_dict["dim"] = dim

        args = Namespace(**args_dict)

        dataset_dir = os.path.join(args.checkpoint_dir, "checkpoints", args.dataset_name)

        bayes_approx = None
        if args.use_mc_dropout:
            bayes_approx = "mcd"
        elif args.use_diffusion_stochastic:
            bayes_approx = "diff_sto"

        exp_name_parts = [
            f"seed_{args.seed}",
            f"datause_{args.data_usage}%",
            f"UC_{args.uncertainty}",
            f"B_{args.final_budget_factor}",
            f"init_LB_{args.local_budget_init}",
            f"LB_{args.local_budget}",
            bayes_approx,
            f"keepglob_{args.keep_global}%" if getattr(args, "keep_global", 0.0) != 0.0 else None,
            f"pt_{args.pretrained_model}",
            f"round_{args.n_stages}"
        ]

        args.dir_checkpoints = os.path.join(dataset_dir, "_".join(filter(None, exp_name_parts)))
        args.exp_name_parts = exp_name_parts
        os.makedirs(args.dir_checkpoints, exist_ok=True)

        with open(os.path.join(args.dir_checkpoints, "args.json"), 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=4, ensure_ascii=False)

        return args