# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from numpy.random import uniform

from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.color import hsv2rgba


def shader_dark_glass(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    shader = nw.new_node(
        Nodes.GlassBSDF,
        input_kwargs={
            "Roughness": 0.0200,
            "IOR": 1.5,
            "Color": hsv2rgba(uniform(0, 1), 0.01, uniform(0.1, 0.2)),
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": shader},
        attrs={"is_active_output": True},
    )


# def apply(obj, selection=None, **kwargs):
#     common.apply(
#         obj,
#         shader_glass,
#         selection,
#     )
