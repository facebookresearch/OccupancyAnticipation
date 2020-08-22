#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from occant_utils.astar_pycpp import pyastar
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, worker_id, use_weighted_graph, scale, niters):
    parent_remote.close()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "plan":
                map_, start, goal, mask, allow_diagonal = data
                if mask == 1 and not use_weighted_graph:
                    path_x, path_y = pyastar.astar_planner(
                        map_, start, goal, allow_diagonal
                    )
                elif mask == 1 and use_weighted_graph:
                    path_x, path_y = pyastar.weighted_astar_planner(
                        map_, start, goal, allow_diagonal, scale, niters,
                    )
                else:
                    path_x, path_y = None, None
                remote.send((path_x, path_y))
            elif cmd == "close":
                remote.close()
                break
    except KeyboardInterrupt:
        print("AStarPlannerVector worker: got KeyboardInterrupt")


class AStarPlannerVector:
    def __init__(self, config):
        self.config = config
        nplanners = config.nplanners
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nplanners)])
        self.ps = [
            Process(
                target=worker,
                args=(
                    work_remote,
                    remote,
                    worker_id,
                    config.use_weighted_graph,
                    config.weight_scale,
                    config.weight_niters,
                ),
            )
            for (work_remote, remote, worker_id) in zip(
                self.work_remotes, self.remotes, range(nplanners)
            )
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def plan_async(self, maps, starts, goals, masks):
        self._assert_not_closed()
        for remote, map_, start, goal, mask in zip(
            self.remotes, maps, starts, goals, masks
        ):
            remote.send(("plan", (map_, start, goal, mask, self.config.allow_diagonal)))
        self.waiting = True

    def plan_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results  # Planned paths

    def plan(self, maps, starts, goals, masks):
        self.plan_async(maps, starts, goals, masks)
        return self.plan_wait()

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))

    def _assert_not_closed(self):
        assert (
            not self.closed
        ), "Trying to operate on an AStarPlannerVector after calling close()"


class AStarPlannerSequential:
    def __init__(self, config):
        self.config = config

    def plan(self, maps, starts, goals, masks):
        paths = []
        for map_, start, goal, mask in zip(maps, starts, goals, masks):
            if mask == 1 and not self.config.use_weighted_graph:
                path_x, path_y = pyastar.astar_planner(
                    map_, start, goal, self.config.allow_diagonal
                )
            elif mask == 1 and self.config.use_weighted_graph:
                path_x, path_y = pyastar.weighted_astar_planner(
                    map_,
                    start,
                    goal,
                    self.config.allow_diagonal,
                    self.config.weight_scale,
                    self.config.weight_niters,
                )
            else:
                path_x, path_y = None, None
            paths.append((path_x, path_y))

        return paths

    def close(self):
        pass

    def _assert_not_closed(self):
        pass
