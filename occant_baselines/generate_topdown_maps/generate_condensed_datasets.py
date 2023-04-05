import os
import os.path as osp

import argparse
import json
import glob
import gzip
import numpy as np
import tqdm


def load_dataset(path):
    with gzip.open(path, "rt") as fp:
        data = json.load(fp)
    return data


def save_dataset(path, data):
    with gzip.open(path, "wt") as fp:
        json.dump(data, fp)


def cluster_scene_episodes(scene_episodes):
    pass


def main(args):

    for split in args.splits:
        per_scene_episodes = {}
        main_json = osp.join(args.dataset_dir, split, f"{split}.json.gz")
        dataset = load_dataset(main_json)
        if len(dataset["episodes"]) == 0:
            per_scene_files = True
        else:
            per_scene_files = False
        if not per_scene_files:
            for ep in dataset["episodes"]:
                scene_id = ep["scene_id"]
                if scene_id not in per_scene_episodes:
                    per_scene_episodes[scene_id] = []
                per_scene_episodes[scene_id].append(ep)
        else:
            ptn = osp.join(args.dataset_dir, split, "content/*.json.gz")
            for scene_file in glob.glob(ptn):
                dataset = load_dataset(scene_file)
                scene_id = dataset["episodes"][0]["scene_id"]
                per_scene_episodes[scene_id] = dataset["episodes"]

        # Find the different navigable heights in each scene
        scenes_to_floor_heights = {}
        for scene_id, scene_data in tqdm.tqdm(per_scene_episodes.items(), desc="Find floor heights"):
            # Identify the number of unique floors in this scene
            floor_heights = []
            for ep in scene_data:
                height = ep["start_position"][1]
                if len(floor_heights) == 0:
                    floor_heights.append(height)
                # Measure height difference from all existing floors
                d2floors = map(lambda x: abs(x - height), floor_heights)
                d2floors = np.array(list(d2floors))
                if not np.any(d2floors < 0.5):
                    floor_heights.append(height)
            # Store this in the dict
            scenes_to_floor_heights[scene_id] = floor_heights

        # Find one episode per floor
        per_scene_episodes_per_floor = {}
        for scene_id, scene_data in per_scene_episodes.items():
            per_scene_episodes_per_floor[scene_id] = {}
            n_eps = 0
            for ep in tqdm.tqdm(scene_data, desc="Processing scene {}".format(scene_id)):
                start_height = ep["start_position"][1]
                floor_heights = scenes_to_floor_heights[scene_id]
                d2floors = map(lambda x: abs(x - start_height), floor_heights)
                d2floors = np.array(list(d2floors))
                floor_idx = np.where(d2floors < 0.5)[0][0].item()
                if floor_idx not in per_scene_episodes_per_floor[scene_id]:
                    per_scene_episodes_per_floor[scene_id][floor_idx] = ep
                    n_eps += 1
            # Spit out stats
            print("Condensed dataset from {} to {}".format(len(scene_data), n_eps))
    
        # Condense episodes and save 
        if per_scene_files:
            save_root = osp.join(args.save_dir, split, "content")
            os.makedirs(save_root, exist_ok=True)
            per_scene_episodes_condensed = {}
            for scene_id, scene_data_per_floor in per_scene_episodes_per_floor.items():
                per_scene_episodes_condensed[scene_id] = []
                for floor_idx, ep in scene_data_per_floor.items():
                    per_scene_episodes_condensed[scene_id].append(ep)
                scene_name = osp.basename(scene_id).split(".")[0]
                save_path = osp.join(save_root, scene_name + ".json.gz")
                save_dataset(save_path, {"episodes": per_scene_episodes_condensed[scene_id]})
            save_path = osp.join(args.save_dir, split, f"{split}.json.gz")
            save_dataset(save_path, {"episodes": {}})
        else:
            save_root = osp.join(args.save_dir, split)
            os.makedirs(save_root, exist_ok=True)
            save_path = osp.join(save_root, f"{split}.json.gz")
            episodes = []
            for scene_id, scene_data_per_floor in per_scene_episodes_per_floor.items():
                for floor_idx, ep in scene_data_per_floor.items():
                    episodes.append(ep)
            save_dataset(save_path, {"episodes": episodes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--splits", type=str, default=["train", "val"], nargs="+")
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)
