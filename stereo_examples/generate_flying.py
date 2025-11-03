# Copyright (c) Princeton University.
# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import argparse
import logging
from pathlib import Path

# ruff: noqa: E402
# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import bpy
import gin
import numpy as np

from infinigen.core import execute_tasks, init, surface
from infinigen.core.placement import camera as camera_place
from infinigen.core.util import blender as butil
from infinigen.core.util import camera as camera_util
from infinigen.core.util import pipeline
from infinigen.core.util.random import random_general as rg
from infinigen.core.constraints.example_solver.solve import Solver # noqa: F401 # needed for indoor configs
from infinigen.core.util.test_utils import (
    import_item,
    load_txt_list,
)
from stereo_examples import flying_lighting

from infinigen_examples import (
    generate_nature,  # noqa: F401 # needed for nature gin configs to be loaded
)

logger = logging.getLogger(__name__)


def remove_mats(obj):
    if not obj.material_slots:
        return
    with butil.SelectObjects(obj):
        for index, slot in reversed(list(enumerate(obj.material_slots))):
            bpy.context.object.active_material_index = index
            bpy.ops.object.material_slot_remove()


def remove_glass_metal(obj, mat_facs):
    if not obj.material_slots:
        return
    with butil.SelectObjects(obj):
        for index, slot in reversed(list(enumerate(obj.material_slots))):
            mat = slot.material
            if mat is None:
                continue
            if (
                "glass" in mat.name in mat.name
                or "metal" in mat.name
                or "medal" in mat.name
                or "mirror" in mat.name
            ):
                bpy.context.object.active_material_index = index
                bpy.ops.object.material_slot_remove()

            if len(obj.material_slots) == 0:
                fac = np.random.choice(mat_facs)
                logging.info(f"adding mat to {obj.name}, no mat")
                fac = np.random.choice(mat_facs)
                if fac != "None":
                    mat_instance = fac()
                    surface.assign_material(obj, mat_instance.generate())



@gin.configurable
def initialize_flying_meshes(
    indoor_paths,
    nature_paths,
    num_objs,
    apply_deform,
    with_materials,
    with_glass,
    use_indoors,
    use_nature,
):
    mesh_facs = []

    if use_indoors:
        mesh_facs = [import_item(path) for path in indoor_paths]

    mat_paths = load_txt_list(Path(__file__).parent / "floating_mats.txt")
    mat_facs = [import_item(path) for path in mat_paths]
    mat_facs.append("None")
    indoor_cutoff = len(mesh_facs)

    if use_nature:
        nature_facs = [import_item(path) for path in nature_paths]
        mesh_facs += nature_facs

    asset_list = {}

    for _ in range(num_objs):
        seed = np.random.randint(1, 2**28)
        i = np.random.randint(0, len(mesh_facs))
        fac = mesh_facs[i](seed)
        asset = fac.spawn_asset(0, loc=(0, 0, -5))

        fac.finalize_assets([asset])
        if i < indoor_cutoff:
            if np.random.uniform() < 0.5:
                remove_mats(asset)
                fac = np.random.choice(mat_facs)
                if fac != "None":
                    mat_instance = fac()
                    surface.assign_material(asset, mat_instance.generate())


            if apply_deform:
                with butil.SelectObjects(asset):
                    deform_types = ["TWIST", "BEND", "TAPER"]
                    random_deform = np.random.choice(deform_types)
                    mod_name = "RandomSimpleDeform"
                    modifier = bpy.context.object.modifiers.new(
                        name=mod_name, type="SIMPLE_DEFORM"
                    )
                    if modifier is not None:
                        modifier.deform_method = random_deform

        # size_penalty = max(np.cbrt((asset.dimensions.x * asset.dimensions.y * asset.dimensions.z)), 1)
        logger.info((asset.name))
        max_dim = max(asset.dimensions.x, asset.dimensions.y, asset.dimensions.z)
        if max_dim != 0:
            normalize_scale = 1 / max(
                asset.dimensions.x, asset.dimensions.y, asset.dimensions.z
            )
        else:
            normalize_scale = 1

        if not with_materials:
            remove_mats(asset)

        if not with_glass:
            remove_glass_metal(asset, mat_facs)

        asset_list[asset] = normalize_scale

    for i in range(10):
        bpy.ops.object.light_add(type="POINT", location=(0, 0, -5))
        light = bpy.context.object
        asset_list[light] = 1

    bpy.ops.object.light_add(type="SUN", align="WORLD", location=(0, 0, 10))
    sun = bpy.context.object
    asset_list[sun] = 1

    return asset_list


@gin.configurable
def resample_flying(
    assets,
    cam_left,
    cam_right,
    W,
    H,
    cx,
    cy,
    fx,
    fy,
    t,
    baseline,
    normalize,
    old_sizing,
    all_dark,
    rgb_bkg,
):
    world_node_tree = bpy.context.scene.world.node_tree
    background_node = world_node_tree.nodes.get("Background")
    mix_node = world_node_tree.nodes.get("Mix")
    rgb_node = world_node_tree.nodes.get("RGB")

    # if all_dark:
    #     background_node.inputs[1].default_value = 0.01
    # else:
    #     background_node.inputs[1].default_value = np.random.uniform(0.05, 0.5)

    if rgb_bkg:
        background_node.inputs[1].default_value = np.random.uniform(0.05, 0.2)
        mix_node.inputs["Factor"].default_value = 1
    else:
        background_node.inputs[1].default_value = np.random.uniform(0.05, 0.2)
        mix_node.inputs["Factor"].default_value = np.random.choice(
            [0, 0.5, 1], p=[0.25, 0.25, 0.5]
        )

    background_node.inputs[1].keyframe_insert(data_path="default_value", frame=t)
    mix_node.inputs["Factor"].keyframe_insert(data_path="default_value", frame=t)

    rgb_node.outputs["Color"].default_value = (
        np.random.uniform(),
        np.random.uniform(),
        np.random.uniform(),
        1,
    )
    rgb_node.outputs["Color"].keyframe_insert(data_path="default_value", frame=t)

    bpy.context.scene.cycles.film_exposure = 1
    # bpy.scene.view_settings.exposure = np.random.uniform(0.5, 3)
    # bpy.scene.view_settings.keyframe_insert(data_path="exposure", frame=t)

    _, cam_y, cam_z = cam_right.location
    cam_right.location = (rg(baseline), cam_y, cam_z)
    #  cam_left.location = (-rg(baseline) / 2, cam_y, cam_z)
    cam_right.keyframe_insert(data_path="location", frame=t)
    cam_left.keyframe_insert(data_path="location", frame=t)

    for asset, normalize_factor in assets.items():
        x = np.random.randint(W)
        y = np.random.randint(H)
        if asset.type == "LIGHT":
            if asset.data.type == "SUN":
                asset.data.color = (
                    np.random.uniform(),
                    np.random.uniform(),
                    np.random.uniform(),
                )
                asset.rotation_euler = np.random.uniform(-np.pi, np.pi, 3)
                asset.data.energy = np.random.uniform(0, 5)
                asset.keyframe_insert(data_path="rotation_euler", frame=t)
                asset.data.keyframe_insert(data_path="energy", frame=t)
                asset.data.keyframe_insert(data_path="color", frame=t)

            else:
                Z = np.random.uniform(*(5, 25))
                X = ((x - cx) / fx) * Z
                Y = ((y - cy) / fy) * Z
                asset.location = (X, Z, Y)
                asset.data.energy = np.random.uniform(0, 3000)
                asset.data.color = (
                    np.random.uniform(),
                    np.random.uniform(),
                    np.random.uniform(),
                )
                asset.keyframe_insert(data_path="location", frame=t)
                asset.data.keyframe_insert(data_path="energy", frame=t)
                asset.data.keyframe_insert(data_path="color", frame=t)

        else:
            Z = np.random.uniform(*(5, 50))
            X = ((x - cx) / fx) * Z
            Y = ((y - cy) / fy) * Z
            asset.location = (X, Z, Y)
            asset.rotation_mode = "XYZ"
            asset.rotation_euler = np.random.uniform(-np.pi, np.pi, 3)

            if old_sizing:
                scale = np.random.uniform(0.4, Z / 10)
            else:
                scale = np.random.normal(Z / 7.5, 0.25)
            if normalize:
                scale *= normalize_factor

            asset.scale = (scale, scale, scale)
            asset.keyframe_insert(data_path="location", frame=t)
            asset.keyframe_insert(data_path="rotation_euler", frame=t)
            asset.keyframe_insert(data_path="scale", frame=t)

            mod = asset.modifiers.get("RandomSimpleDeform")
            if mod is None:
                continue
            if np.random.uniform() < 0.5:
                mod.angle = 0
            else:
                mod.angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            mod.keyframe_insert("angle", frame=t)

        # size_penalty = max(np.cbrt(asset.dimensions.x * asset.dimensions.y * asset.dimensions.x), 1)
        # asset.scale = (scale/size_penalty, scale/size_penalty, scale/size_penalty)


@gin.configurable
def compose_indoors(output_folder: Path, scene_seed: int, **overrides):
    p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)

    bkgd_nodes = bpy.context.scene.world.node_tree.nodes
    for node in bkgd_nodes:
        bkgd_nodes.remove(node)

    p.run_stage("sky_lighting", flying_lighting.add_lighting, use_chance=False)

    assets = initialize_flying_meshes()

    camera_rig = camera_place.spawn_camera_rigs()[0]
    camera_rig.rotation_euler = (1.5708, 0, 0)
    W = 1280
    H = 720

    cam_left = camera_rig.children[0]
    cam_right = camera_rig.children[1]
    camera_place.adjust_camera_sensor(cam_left)
    camera_place.adjust_camera_sensor(cam_right)
    K = camera_util.get_calibration_matrix_K_from_blender(cam_left.data)
    cx, cy = W // 2, H // 2
    fx, fy = K[0][0], K[1][1]

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            distance = (
                cam_left.matrix_world.to_translation()
                - obj.matrix_world.to_translation()
            ).length
            if distance < 1:
                logger.info(f"Hiding {obj.name}")
                obj.hide_render = True

    for i in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        resample_flying(assets, cam_left, cam_right, W, H, cx, cy, fx, fy, i)


def main(args):
    scene_seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(
        configs=["base_indoors.gin"] + args.configs,
        overrides=args.overrides,
        config_folders=[
            "infinigen_examples/configs_indoor",
            "infinigen_examples/configs_nature",
            "stereo_examples/configs_floating",
            "stereo_examples/configs_flying",
        ],
    )

    execute_tasks.main(
        compose_scene_func=compose_indoors,
        populate_scene_func=None,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task=args.task,
        task_uniqname=args.task_uniqname,
        scene_seed=scene_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--input_folder", type=Path, default=None)
    parser.add_argument(
        "-s", "--seed", default=None, help="The seed used to generate the scene"
    )
    parser.add_argument(
        "-t",
        "--task",
        nargs="+",
        default=["coarse"],
        choices=[
            "coarse",
            "populate",
            "fine_terrain",
            "ground_truth",
            "render",
            "mesh_save",
            "export",
        ],
    )
    parser.add_argument(
        "-g",
        "--configs",
        nargs="+",
        default=["base"],
        help="Set of config files for gin (separated by spaces) "
        "e.g. --gin_config file1 file2 (exclude .gin from path)",
    )
    parser.add_argument(
        "-p",
        "--overrides",
        nargs="+",
        default=[],
        help="Parameter settings that override config defaults "
        "e.g. --gin_param module_1.a=2 module_2.b=3",
    )
    parser.add_argument("--task_uniqname", type=str, default=None)
    parser.add_argument("-d", "--debug", type=str, nargs="*", default=None)

    args = init.parse_args_blender(parser)
    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("infinigen"):
                continue
            if len(args.debug) == 0 or any(name.endswith(x) for x in args.debug):
                logging.getLogger(name).setLevel(logging.DEBUG)

    main(args)
