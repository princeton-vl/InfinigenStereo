# What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?

We introduce **WMGStereo**, a procedural dataset generator specifically optimized for zero-shot stereo matching performance. Using our generator, we create and release WMGStereo-150k, a new training dataset for stereo matching.

If you find WMGStereo useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2504.16930">
        What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?
    </a>
</h3>
<p align="center">
    <a href="https://david-yan1.github.io">David Yan</a>, 
    <a href="https://araistrick.github.io">Alexander Raistrick</a>, 
    <a href="https://www.cs.princeton.edu/~jiadeng/">Jia Deng</a><br/>
</p>

```
@misc{yan2025proceduraldatasetgenerationzeroshot,
      title={What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?}, 
      author={David Yan and Alexander Raistrick and Jia Deng},
      year={2025},
      eprint={2504.16930},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.16930}, 
}
```

<img src="imgs/sample.jpg" width='1400'>


## Install

To populate the Infinigen submodule, run

```
git submodule init
git submodule update
```

Symlink or copy the stereo modification code by running

```
ln -s stereo_examples infinigen-module/stereo_examples
```

Then, install Infinigen by running 

```
conda create --name infinigen python=3.11
conda activate infinigen

cd infinigen-submodule
pip install -e ".[dev,terrain,vis]"
```

## Generating new data

Inside the `infinigen-submodule` directory, you can run the following commands to generate scenes. To modify data generation settings, the main relevant configs and driver scripts are in `stereo_examples`.

Generate indoor scenes:

``` 
python -m infinigen.datagen.manage_jobs --output_folder {OUTPUT_FOLDER} --num_scenes {N} --configs singleroom trailer_video floating_solve floating --pipeline_configs local_256GB.gin stereo blender_gt.gin indoor_background_configs.gin --pipeline_overrides get_cmd.driver_script=stereo_examples.generate_floating iterate_scene_tasks.n_camera_rigs=20 iterate_scene_tasks.n_subcams=2 --overrides compose_indoors.animate_cameras_enabled=False render_image.use_dof=False camera.spawn_camera_rigs.n_camera_rigs=20 compute_base_views.min_candidates_ratio=2 compose_indoors.restrict_single_supported_roomtype=True
```

Generate dense floating/flying scenes:
```
python -m infinigen.datagen.manage_jobs --output_folder {OUTPUT_FOLDER} --num_scenes {N} --wandb_mode offline --configs flying.gin --pipeline_configs local_256GB.gin stereo_video.gin blender_gt.gin indoor_background_configs.gin --pipeline_overrides get_cmd.driver_script=stereo_examples.generate_flying iterate_scene_tasks.frame_range=[1,200] iterate_scene_tasks.view_block_size=1000 iterate_scene_tasks.cam_block_size=25 --overrides compose_indoors.animate_cameras_enabled=False render_image.use_dof=False 
```

Generate nature scenes:

```
python -m infinigen.datagen.manage_jobs  --output_folder {OUTPUT_FOLDER} --num_scenes {N} --configs high_quality_terrain.gin noisy_video.gin nature_stereo --pipeline_configs local_256GB stereo_video.gin cuda_terrain blender_gt.gin --pipeline_overrides get_cmd.driver_script=stereo_examples.generate_nature iterate_scene_tasks.frame_range=[1,50] iterate_scene_tasks.view_block_size=1000 iterate_scene_tasks.cam_block_size=25 --warmup_sec 2000 --cleanup big_files
```

The experiments/data in the paper were generated wih an older version of Infinigen. For reproducibility, we provide our code in `infinigen-old-exp`. To generate data, follow installation instructions inside `infinigen-old-exp/docs/Installation.md` and run the same commands from `infinigen-old-exp`.



## WMGStereo Dataset

Our dataset is now available on <a href=" https://huggingface.co/datasets/pvl-lab/WMGStereo"> HuggingFace</a>. 
You can download it with the command 

```
pip install huggingface-cli
huggingface-cli download pvl-lab/WMGStereo --repo-type dataset
```

The dataset file structure is as follows:

```
.
└── WMGStereo/
    ├── indoor/
    │   └── seed_num/
    │       └── frames/
    │           ├── Image/
    │           │   ├── camera_0
    │           │   └── camera_1
    │           ├── camview/
    │           │   ├── camera_0
    │           │   └── camera_1
    │           ├── disparity/
    │           │   └── camera_0
    │           ├── occ_mask/
    │           │   └── camera_0
    │           └── sky_mask/
    │               └── camera_0
    ├── flying/
    │   └── ...
    └── nature/
        └── ...
```

Camera 0 and 1 correspond to left and right camera frames, respectively. 
We provide disparity, occlusion, sky-region masks for the left camera.
`camview` contains `.npz` files that contain a dictionary with indices `K`, `T`, `HW`, corresponding to calibration, translation, and resolution matrices.