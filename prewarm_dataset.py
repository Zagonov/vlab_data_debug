from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.smolvla2.configuration_smolvla2 import SmolVLA2Config

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    cfg.validate()

    print("Starting dataset preprocessing...")
    dataset = make_dataset(cfg)

    print("Dataset preprocessing finished.")
    print("num_frames:", dataset.num_frames)
    print("num_episodes:", dataset.num_episodes)


if __name__ == "__main__":
    main()

