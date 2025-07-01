# Procedural Dataset Generation for Zero-Shot Stereo Matching

We introduce **Infinigen-Stereo**, a procedural dataset generator specifically optimized for zero-shot stereo matching performance. Using our generator, we create and release Infinigen-Stereo-150k, a new training dataset for stereo matching.

If you find Infinigen-Stereo useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2504.16930">
        Procedural Dataset Generation for Zero-Shot Stereo Matching
    </a>
</h3>
<p align="center">
    <a href="https://david-yan1.github.io">David Yan</a>, 
    <a href="https://araistrick.github.io">Alexander Raistrick</a>, 
    <a href="https://www.cs.princeton.edu/~jiadeng/">Jia Deng</a><br/>
</p>

```
@misc{yan2025proceduraldatasetgenerationzeroshot,
      title={Procedural Dataset Generation for Zero-Shot Stereo Matching}, 
      author={David Yan and Alexander Raistrick and Jia Deng},
      year={2025},
      eprint={2504.16930},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.16930}, 
}
```

<img src="imgs/sample.jpg" width='1400'>


## Generator Code

Coming Soon!

## InfinigenStereo Dataset

Our dataset is now available on <a href=" https://huggingface.co/datasets/pvl-lab/InfinigenStereo"> HuggingFace</a>. 
You can download it with the command 

```
pip install huggingface-cli
huggingface-cli download pvl-lab/InfinigenStereo --repo-type dataset
```

The dataset file structure is as follows:

```
.
└── InfinigenStereo/
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


