#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The script uses the locations sampled by pointnav task as the starting point
and samples an exploration episode.
"""

import os
import pdb
import gzip
import json
import tqdm
import random
import argparse
import numpy as np
import subprocess as sp


def main(args):
    for split in args.splits:
        print("=====> Creating dataset: {} split".format(split))
        data_path = f"{args.pointnav_dataset_path}/{split}/{split}.json.gz"
        with gzip.open(data_path, "rt") as fp:
            data = json.load(fp)

        if data["episodes"] == []:
            # Save empty data to ep_save_path
            data_save_path = f"{args.save_dataset_path}/{split}"
            sp.call(f"mkdir -p {data_save_path}", shell=True)

            ep_save_path = f"{data_save_path}/{split}.json.gz"
            with gzip.open(ep_save_path, "wt") as fp:
                json.dump({"episodes": []}, fp)

            # Process scene specific data and store in content/
            data_dir = f"{args.pointnav_dataset_path}/{split}/content/"
            scenes = (
                sp.check_output(f"ls {data_dir}", shell=True)
                .decode("UTF-8")
                .split("\n")[:-1]
            )

            data_save_path = f"{args.save_dataset_path}/{split}/content/"
            sp.call(f"mkdir -p {data_save_path}", shell=True)

            # Process each scene individually
            print("========= Processing individual scenes ==========")
            for scene_itr in tqdm.tqdm(range(len(scenes))):
                scene = scenes[scene_itr]

                # Read scene data
                scene_path = os.path.join(data_dir, scene)
                with gzip.open(scene_path, "rt") as fp:
                    scene_episode_data = json.load(fp)["episodes"]

                # Sample pose references for each episode
                episode_data_all = []
                for ep in scene_episode_data:
                    scene_id = ep["scene_id"]
                    start_position = ep["start_position"]
                    episode_data = {
                        "episode_id": ep["episode_id"],
                        "scene_id": scene_id,
                        "start_position": ep["start_position"],
                        "start_rotation": ep["start_rotation"],
                    }
                    episode_data_all.append(episode_data)

                # Save data to scene_save_path
                scene_save_path = os.path.join(data_save_path, scene)
                with gzip.open(scene_save_path, "wt") as fp:
                    json.dump({"episodes": episode_data_all}, fp)

        else:
            # Sample pose references for each episode
            episode_data_all = []
            print("========= Processing episodes ==========")
            for ep_itr in tqdm.tqdm(range(len(data["episodes"]))):
                ep = data["episodes"][ep_itr]
                scene_id = ep["scene_id"]
                start_position = ep["start_position"]
                episode_data = {
                    "episode_id": ep["episode_id"],
                    "scene_id": scene_id,
                    "start_position": ep["start_position"],
                    "start_rotation": ep["start_rotation"],
                }
                episode_data_all.append(episode_data)

            # Save data to ep_save_path
            data_save_path = f"{args.save_dataset_path}/{split}"
            sp.call(f"mkdir -p {data_save_path}", shell=True)

            ep_save_path = f"{data_save_path}/{split}.json.gz"
            with gzip.open(ep_save_path, "wt") as fp:
                json.dump({"episodes": episode_data_all}, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointnav_dataset_path", type=str, default="")
    parser.add_argument("--save_dataset_path", type=str, default="")
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val", "test"]
    )

    args = parser.parse_args()

    main(args)
