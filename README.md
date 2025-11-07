# Tether: Autonomous Play with Correspondence-Driven Trajectory Warping

<div align="center">

[[Website]]
[[arXiv]]
[[PDF]]

[William Liang](https://willjhliang.github.io), [Sam Wang](https://samuelwang23.github.io/), [Hung-Ju Wang](https://www.linkedin.com/in/hungju-wang),<br>
[Osbert Bastani](https://obastani.github.io/), [Yecheng Jason Ma<sup>†</sup>](https://jasonma2016.github.io/), [Dinesh Jayaraman<sup>†</sup>](https://www.seas.upenn.edu/~dineshj/)

University of Pennsylvania

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://github.com/tether-research/tether)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/tether-research/tether)](https://github.com/tether-research/tether/blob/main/LICENSE)

______________________________________________________________________

</div>

The ability to conduct and learn from self-directed interaction and experience is a central challenge in robotics, offering a scalable alternative to labor-intensive human demonstrations. However, realizing such "play" requires (1) a policy robust to diverse, potentially out-of-distribution environment states, and (2) a procedure that continuously produces useful, task-directed robot experience. To address these challenges, we introduce Tether, a method for autonomous play with two key contributions. First, we design a novel non-parametric policy that leverages strong visual priors for extreme generalization: given two-view images, it identifies semantic correspondences to warp demonstration trajectories into new scenes. We show that this design is robust to significant spatial and semantic variations of the environment, such as dramatic positional differences and unseen objects. We then deploy this policy for autonomous multi-task play in the real world via a continuous cycle of task selection, execution, evaluation, and improvement, guided by the visual understanding capabilities of vision-language models. This procedure generates diverse, high-quality datasets with minimal human intervention. In a household-like multi-object setup, our method is among the first to perform many hours of autonomous real-world play, producing a stream of data that consistently improves downstream policy performance over time. Ultimately, Tether yields over 1000 expert-level trajectories and trains policies competitive with those learned from human-collected demonstrations.

# Installation
The following instructions will install everything three conda environments: one main environment for the tether code, and two conda environments for running GeoAware and Mast3r. We have tested on Ubuntu 20.04.

1. Create the main Conda environment with:
    ```
    conda create -n tether python=3.10
    conda activate tether
    pip install -r requirements.txt
    ```

2. Follow the instructions [here](https://github.com/Junyi42/GeoAware-SC?tab=readme-ov-file#environment-setup) to setup the conda environment for GeoAware. 

3. Follow the instructions [here](https://github.com/naver/mast3r) to setup the conda environment for Mast3r.
   
4. For real world experiments, we deploy Tether with [Eva](https://github.com/willjhliang/eva), our Franka Infrastructure code. Installation instructions for Eva can be found (here)[https://github.com/willjhliang/eva?tab=readme-ov-file#installation].

# Usage
1. Set your Gemini API key in `conf/config.yaml` under the `api_key_smart` and `api_key_fast` fields.

2. Collect the initial set of demonstrations for your target tasks. Place your demonstrations for the given setting (eg. "real") under the `./data_<SETTING_NAME>/demos` folder.

3. Edit the demo_names list in the `./conf/setting/<SETTING_NAME>.yaml` configuration to match your demonstration set. The format of this list is: `<name of the subdirectory in demo folder>`: `<desired natural instruction for Gemini action planning and success evaluation>`

4. In `./conf/setting/real.yaml`, modify the camera parameters to be the ZED camera serial numbers in your setup. You can find the serial numbers for your cameras using [these instructions](https://support.stereolabs.com/hc/en-us/articles/19540095753111-How-can-I-get-the-serial-number-of-my-camera).

5. Adjust the oob_bounds parameters to the desired workspace in your scene. If the robot exceeds the desired workspace parameters during the execution of a trajectory, it will stop.

# Running the Policy

1. In the respective conda environments from the last section, start the servers for GeoAware and Mast3r by running `serve_geoaware.py` and `server_mast3r.py`. Wait for the servers to both print Serving ... before proceeding.

2. Follow the instructions for starting up the Eva server and runner [here](https://github.com/willjhliang/eva?tab=readme-ov-file#startup).

3. To generate a single rollout, run `python runner.py mode=single action=<Name of action from setting config>`. This will select a random demo for your specified action and warp it for the current scene. The rollout data will be saved under `data_{setting}/runs/<run_name>/rollouts_single`.

4. To run the autonomous play procedure, run `python runner.py mode=cycle`. This will begin by preprocessing the demos into the action library. It then will will run a cycle of action selection with the VLM, executing the selected action using Tether, and success evaluation using the VLM. The rollout data will be saved under `data_{setting}/runs/<run_name>/rollouts_cycle`.

## Acknowledgements
We thank the following open-sourced projects:
* We calcualte correspondences using the [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC) and [mast3r](https://github.com/naver/mast3r) projects.
* Our Franka hardware setup is from [DROID](https://github.com/Junyi42/GeoAware-SC/tree/b20fab9c2d4f686536be9db34d1fb2079240fdd5).
* Our deployment infrastructure,[Eva](https://github.com/willjhliang/eva), builds on [DROID](https://droid-dataset.github.io/droid/docs/software-setup)'s software setup.

# License
This codebase is released under [MIT License](LICENSE).

## Citation
If you find our work useful, please consider citing us!
```bibtex
@misc{liang2025tether,
    title={Tether: Autonomous Play with Correspondence-Driven Trajectory Warping},
    author  = {William Liang and Sam Wang and Hungju Wang and Osbert Bastani and Jason Ma and Dinesh Jayaraman}
    year={2025},
}
```
