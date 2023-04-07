#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import math
import numpy as np

from habitat_extensions import get_extended_config as get_task_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.PYT_RANDOM_SEED = 123
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/exploration_gibson.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "occant_exp"
_C.ENV_NAME = "ExpRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.SIMULATOR_GPU_IDS = []  # Assign specific GPUs to simulator
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = -1
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.EVAL_PREV_CKPT_ID = -1  # The evaluation starts at (this value + 1)th ckpt
_C.NUM_PROCESSES = 36
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_EPISODES = 20000
_C.T_EXP = 1000
_C.LOG_FILE = "train.log"
_C.SAVE_STATISTICS_FLAG = False
_C.CHECKPOINT_INTERVAL = 30
# PointNav specific config
_C.T_MAX = 500
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
# PointNav specific config
_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.SUCCESS_MEASURE = "spl"
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 4
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 1e-3
_C.RL.PPO.local_entropy_coef = 1e-2
_C.RL.PPO.lr = 2.5e-4
_C.RL.PPO.local_policy_lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.use_gae = True
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.loss_stats_window_size = 100
_C.RL.PPO.local_reward_scale = 0.1
_C.RL.PPO.global_reward_scale = 1e-4
_C.RL.PPO.num_local_steps = 25
_C.RL.PPO.num_global_steps = 20
# -----------------------------------------------------------------------------
# ACTIVE NEURAL SLAM (ANS)
# -----------------------------------------------------------------------------
_C.RL.ANS = CN()
_C.RL.ANS.pyt_random_seed = 123
_C.RL.ANS.planning_step = 0.50  # max distance of local goal from current position
_C.RL.ANS.goal_success_radius = 0.2  # success threshold for reaching a goal
_C.RL.ANS.goal_interval = 25  # goal sampling interval for global policy
_C.RL.ANS.thresh_explored = 0.6  # threshold to classify a cell as explored
_C.RL.ANS.thresh_obstacle = 0.6  # threshold to classify a cell as an obstacle
_C.RL.ANS.overall_map_size = 961  # world map size M
_C.RL.ANS.reward_type = "area_seen"  # Can be area_seen / map_accuracy
_C.RL.ANS.local_slack_reward = -0.3
_C.RL.ANS.local_collision_reward = -1.0
_C.RL.ANS.stop_action_id = 3
_C.RL.ANS.forward_action_id = 0
_C.RL.ANS.left_action_id = 1
_C.RL.ANS.image_scale_hw = [128, 128]
_C.RL.ANS.model_path = ""
_C.RL.ANS.recovery_heuristic = "random_explored_towards_goal"
_C.RL.ANS.crop_map_for_planning = True
# =============================================================================
# Mapper
# =============================================================================
_C.RL.ANS.MAPPER = CN()
_C.RL.ANS.MAPPER.lr = 1e-3
_C.RL.ANS.MAPPER.eps = 1e-5
_C.RL.ANS.MAPPER.max_grad_norm = 0.5
_C.RL.ANS.MAPPER.num_mapper_steps = 100  # number of steps per mapper update
_C.RL.ANS.MAPPER.map_size = 101  # V
_C.RL.ANS.MAPPER.map_scale = 0.05  # s in meters
_C.RL.ANS.MAPPER.projection_unit = "none"
_C.RL.ANS.MAPPER.pose_loss_coef = 30.0
_C.RL.ANS.MAPPER.detach_map = False
_C.RL.ANS.MAPPER.registration_type = "moving_average"
_C.RL.ANS.MAPPER.map_registration_momentum = 0.9
_C.RL.ANS.MAPPER.thresh_explored = 0.6  # threshold to classify a cell as explored
_C.RL.ANS.MAPPER.thresh_entropy = (
    0.5  # entropy threshold to classify a cell as confident
)
_C.RL.ANS.MAPPER.freeze_projection_unit = False
_C.RL.ANS.MAPPER.pose_predictor_inputs = ["ego_map"]
_C.RL.ANS.MAPPER.n_pose_layers = 1
_C.RL.ANS.MAPPER.n_ensemble_layers = 1
_C.RL.ANS.MAPPER.ignore_pose_estimator = False
_C.RL.ANS.MAPPER.label_id = "ego_map_gt_anticipated"
_C.RL.ANS.MAPPER.use_data_parallel = False
_C.RL.ANS.MAPPER.gpu_ids = []  # Set the GPUs for data parallel if necessary
_C.RL.ANS.MAPPER.num_update_batches = 50
_C.RL.ANS.MAPPER.replay_size = 100000
_C.RL.ANS.MAPPER.map_batch_size = 400
# Image normalization
_C.RL.ANS.MAPPER.NORMALIZATION = CN()
_C.RL.ANS.MAPPER.NORMALIZATION.img_mean = [0.485, 0.456, 0.406]
_C.RL.ANS.MAPPER.NORMALIZATION.img_std = [0.229, 0.224, 0.225]
# Image scaling
_C.RL.ANS.MAPPER.image_scale_hw = [128, 128]

# =============================================================================
# Occupancy anticipator
# =============================================================================
_C.RL.ANS.OCCUPANCY_ANTICIPATOR = CN()
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.pyt_random_seed = 123
# Type of model to use
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.type = "occant_depth"

# =========== GP_ANTICIPATION specific options ============
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION = CN()
# Model capacity factor for custom UNet
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.unet_nsf = 16
# Freeze image features?
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.freeze_features = False
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.nclasses = 2
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.resnet_type = "resnet18"
# OccAnt RGB specific hyperparameters
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.detach_depth_proj = False
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.pretrained_depth_proj_model = ""
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.freeze_depth_proj_model = False
# Normalization options for anticipation output
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.OUTPUT_NORMALIZATION = CN()
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 = (
    "sigmoid"
)
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 = (
    "sigmoid"
)
# Wall occupancy option
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.wall_fov = 120.0

# ====== EGO_PROJECTION specific options =======
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION = CN()
# Output map to project egocentric depth anticipation
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.local_map_shape = (2, 101, 101)
# Gridcell size for map in meters
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.map_scale = 0.05
# Minimum and maximum depth value used to scale results between 0.0 to 1.0
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.min_depth = 0.0
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.max_depth = 10.0
# Used to truncate inputs that are farther than a certain distance away
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.truncate_depth = 3.25
# Field of view of expanded image width
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.hfov = 90
# Field of view of expanded image height (no expansion in height)
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.vfov = 90
# Tilt angle of the camera (in degrees) --- upward tilt is positive, zero tilt is forward
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.tilt = 0
# Camera height (in meters)
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.camera_height = 1.25
# Height thresholds to determine obstacles, free-space (in meters)
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.height_thresholds = [0.2, 1.5]
# Size of image feature / size of depth values - must be set to 1.0 here
_C.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION.K = 1.0

# =============================================================================
# Global policy
# =============================================================================
_C.RL.ANS.GLOBAL_POLICY = CN()
_C.RL.ANS.GLOBAL_POLICY.map_size = 240  # global policy input size G
_C.RL.ANS.GLOBAL_POLICY.use_data_parallel = False  # global policy input size G
_C.RL.ANS.GLOBAL_POLICY.gpu_ids = []  # global policy input size G
# =============================================================================
# Local policy
# =============================================================================
_C.RL.ANS.LOCAL_POLICY = CN()
_C.RL.ANS.LOCAL_POLICY.use_heuristic_policy = False
_C.RL.ANS.LOCAL_POLICY.deterministic_flag = False
_C.RL.ANS.LOCAL_POLICY.learning_algorithm = "rl"  # rl / il
_C.RL.ANS.LOCAL_POLICY.nactions = 3
_C.RL.ANS.LOCAL_POLICY.hidden_size = 256
_C.RL.ANS.LOCAL_POLICY.image_scale_hw = [128, 128]
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS = CN()
# Distance bucketed embedding
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.DISTANCE = CN()
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.DISTANCE.min = 0.0625
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.DISTANCE.max = 4
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.DISTANCE.count = 6
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.DISTANCE.dim = 32
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.DISTANCE.use_log_scale = True
# Angle bucketed embedding
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.ANGLE = CN()
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.ANGLE.min = -math.pi
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.ANGLE.max = math.pi
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.ANGLE.count = 72
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.ANGLE.dim = 32
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.ANGLE.use_log_scale = False
# Time bucketed embedding
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.TIME = CN()
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.TIME.min = 0
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.TIME.max = 1001
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.TIME.count = 30
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.TIME.dim = 32
_C.RL.ANS.LOCAL_POLICY.EMBEDDING_BUCKETS.TIME.use_log_scale = False
# Image normalization
_C.RL.ANS.LOCAL_POLICY.NORMALIZATION = CN()
_C.RL.ANS.LOCAL_POLICY.NORMALIZATION.img_mean = [0.485, 0.456, 0.406]
_C.RL.ANS.LOCAL_POLICY.NORMALIZATION.img_std = [0.229, 0.224, 0.225]
# Agent dynamics information (for imitation action selection)
_C.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS = CN()
_C.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.forward_step = 0.25
_C.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle = 10.0

# =============================================================================
# Planner
# =============================================================================
_C.RL.ANS.PLANNER = CN()
_C.RL.ANS.PLANNER.nplanners = 36  # Same as the number of processes
_C.RL.ANS.PLANNER.allow_diagonal = True  # planning diagonally
# local region around the agent / goal that is set to free space when either
# are classified occupied
_C.RL.ANS.PLANNER.local_free_size = 0.25
# Assign weights to graph based on proximity to obstacles?
_C.RL.ANS.PLANNER.use_weighted_graph = False
# Weight factors
_C.RL.ANS.PLANNER.weight_scale = 4.0
_C.RL.ANS.PLANNER.weight_niters = 1


def get_config(
    config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
