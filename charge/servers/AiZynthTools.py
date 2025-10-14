################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################


import logging

try:
    from aizynthfinder.aizynthfinder import AiZynthFinder
    from aizynthfinder.utils.logging import setup_logger
    setup_logger(console_level=logging.INFO)
except ImportError:
    raise ImportError("Please install the aizynthfinder package to use this module.")
import json

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from loguru import logger
from charge.servers.SMILES_utils import verify_smiles

@dataclass
class Node:
    node_id: int
    smiles: str
    children: List[int]  # List of node_ids
    is_root: bool = False
    is_leaf: bool = False
    parent_id: Optional[int] = None
    parent_reaction: Optional[str] = None
    parent_smiles: Optional[str] = None
    purchasable: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        """Return a JSON string representation of the Node."""
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        """Make print(Node(...)) return the JSON representation."""
        return self.to_json()


class ReactionPath:
    def __init__(self, route):
        self.route = route
        self.nodes: Dict[int, Node] = {}
        self.num_nodes = 0
        self._build_path()
        self.leaf_nodes = [
            node_id for node_id, node in self.nodes.items() if node.is_leaf
        ]

    def _build_path(self):

        self.root = Node(
            node_id=0, smiles=self.route["smiles"], children=[], is_root=True
        )
        self.nodes[0] = self.root
        self.num_nodes += 1
        reaction_node = self.route["children"][0]
        if "children" in self.route:
            self._add_children(self.root, reaction_node, reaction_node["children"])

    def _add_children(self, parent_node, reaction, children):

        for child in children:
            if child["type"] == "mol":
                child_node = Node(
                    node_id=self.num_nodes,
                    smiles=child["smiles"],
                    children=[],
                    parent_id=parent_node.node_id,
                    parent_reaction=reaction["smiles"],
                    parent_smiles=parent_node.smiles,
                    purchasable=child.get("in_stock", None),
                )
                parent_node.children.append(self.num_nodes)
                self.nodes[self.num_nodes] = child_node
                self.num_nodes += 1
                if "children" in child:
                    reaction_node = child["children"][0]
                    self._add_children(
                        child_node, reaction_node, reaction_node["children"]
                    )
                else:
                    child_node.is_leaf = True

    def to_json(self):
        return json.dumps(
            self.nodes,
            default=lambda o: asdict(o) if isinstance(o, Node) else str(o),
            indent=2,
        )

    def return_nodes(self) -> Dict[int, Node]:
        return self.nodes


class RetroPlanner:
    finder = None

    def __init__(self, configfile="config.yml", stock_name="zinc", policy_name="uspto"):
        RetroPlanner.finder = AiZynthFinder(configfile=configfile)
        RetroPlanner.finder.stock.select(stock_name)
        RetroPlanner.finder.expansion_policy.select(policy_name)
        RetroPlanner.finder.filter_policy.select(policy_name)

    @staticmethod
    def initialize(configfile="config.yml", stock_name="zinc", policy_name="uspto"):
        if RetroPlanner.finder is None:
            RetroPlanner.finder = AiZynthFinder(configfile=configfile)
            RetroPlanner.finder.stock.select(stock_name)
            RetroPlanner.finder.expansion_policy.select(policy_name)
            RetroPlanner.finder.filter_policy.select(policy_name)

    @staticmethod
    def plan(smiles):
        if RetroPlanner.finder is None:
            raise ValueError("RetroPlanner not initialized. Call initialize() first.")
        RetroPlanner.finder.target_smiles = smiles
        RetroPlanner.finder.tree_search(show_progress=False)
        RetroPlanner.finder.build_routes()
        stats = RetroPlanner.finder.extract_statistics()
        route_dicts = RetroPlanner.finder.routes.make_dicts()
        return RetroPlanner.finder.tree, stats, route_dicts


def is_molecule_synthesizable(smiles: str) -> bool:
    """Checks if a given molecule is synthesizable. First checks if it is
    available in a stock database, otherwise runs a retrosynthesis to see if a
    synthesis route can be found.
    Args:
        smiles (str): The SMILES string of the molecule to check.

    Returns:
        bool: True if the molecule is synthesizable, False otherwise.

    Raises:
        ValueError:  If the molecule is not valid.
    """
    logger.info(f"Checking if molecule {smiles} is synthesizable.")

    if not verify_smiles(smiles):
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if RetroPlanner.finder is None:
        RetroPlanner.initialize()
    assert RetroPlanner.finder is not None
    RetroPlanner.finder.target_smiles = smiles

    tree, stats, routes = RetroPlanner.plan(smiles)
    if len(routes) == 0:
        return False
    for route in routes:
        path = ReactionPath(route=route)
        # check if all leaf nodes are purchasable
        all_purchasable = all(
            path.nodes[node_id].purchasable is True for node_id in path.leaf_nodes
        )
        if all_purchasable:
            return True
    return False

def find_synthesis_routes(smiles: str) -> list[dict]:
    """
    Find synthesis routes for synthesizing a target molecule.

    Args:
        smiles (str): the target molecule in SMILES representation.
    Returns:
        list[dict]: a list of synthesis routes, each of which is a reaction tree in json/dict format.
    Raises:
        ValueError:  If the molecule is not valid.
    """
    logger.info(f"Find a synthesis route for molecule {smiles}.")

    if not verify_smiles(smiles):
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if RetroPlanner.finder is None:
        RetroPlanner.initialize()
    assert RetroPlanner.finder is not None
    RetroPlanner.finder.target_smiles = smiles

    _, _, routes = RetroPlanner.plan(smiles)
    return routes
