# PhyWorldBench 🚀

**A Comprehensive Evaluation of Physical Realism in Text‑to‑Video Models**  
[arXiv Paper (2507.13428)](https://arxiv.org/abs/2507.13428) • [NVIDIA Research Page](https://research.nvidia.com/labs/dir/phyworldbench/) • [Dataset 🤗](https://huggingface.co/datasets/ashwin333/phyworldbench)  
[Jing Gu](https://g-jing.github.io/) • [Xian Liu](https://alvinliu0.github.io/) • [Yu Zeng](https://zengxianyu.github.io/) • [Ashwin Nagarajan](https://github.com/ASHWIN333) • [Fangrui Zhu](https://fangruizhu.github.io/) • [Daniel Hong](https://github.com/danielh-hong) • [Yue Fan](https://yfan.site/) • [Qianqi Yan](https://jackie-2000.github.io/) • [Kaiwen Zhou](https://kevinz-01.github.io/) • [Ming‑Yu Liu](https://mingyuliu.net/) • [Xin Eric Wang](https://eric-xw.github.io/)


![Physics-Simulation](https://img.shields.io/badge/Physics-Simulation-blue)
![Benchmark-Evaluation](https://img.shields.io/badge/Benchmark-Evaluation-blue)
![Text2Video](https://img.shields.io/badge/Text2Video-Evaluation-green)



This repository contains the official evaluation process and data for "PhyWorldBench: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models".

<p align="center">
    <img src="assets/benchmark_overview.png" width="100%"> <br>
    <em>The benchmark follows a structured design with 10 main physics categories, each divided into 5 subcategories, capturing different aspects of physical phenomena.</em>
</p>

This repository contains a pipeline for evaluating AI-generated videos against physics-based standards using vision-language models on Azure platform. The pipeline consists of three main components:

1. Video Frame Sampling
2. Model Evaluation
3. Results Analysis

## Setup

1. Install the required dependencies:
```bash
pip install openai tqdm opencv-python
```

2. Set up your Azure OpenAI API credentials in `evaluate_videos.py`:
> **Note:** If you are using models other than GPT on Azure, or a different API provider, you may need to modify the API setup section in `evaluate_videos.py` accordingly.
```python
api_key = "YOUR_API_KEY"
api_base = "YOUR_API_BASE"
api_version = "YOUR_API_VERSION"
deployment_name = "YOUR_DEPLOYMENT_NAME"
```

## Pipeline Components

### 1. Video Generation and Frame Sampling

#### 1.1 Generate Videos
First, you need to generate videos based on the prompts in `prompt_index_to_prompt.json`. This file contains three versions of each prompt:
- Simple version (e.g., "001-1"): Basic description of the scene
- Detailed version (e.g., "001-2"): More specific description of the physics involved
- Rich version (e.g., "001-3"): Detailed scene description with visual elements

The generated videos should be placed in the `videos` directory, with filenames matching the prompt indices (e.g., `001-1.mp4`, `001-2.mp4`, etc.).

#### 1.2 Sample Frames (`sample_video_frames.py`)

This script samples k frames from your generated videos and saves them to a subfolder.

Usage:
```bash
# Basic usage with required arguments
python sample_video_frames.py --source_folder videos --k 8

# Specify custom output folder name
python sample_video_frames.py --source_folder videos --k 8 --output_folder custom_frames
```

Command-line arguments:
- `--source_folder`: Path to the source folder containing videos (required)
- `--k`: Number of frames to sample from each video (required)
- `--output_folder`: Name of the output subfolder (default: "sampled_frames")

The script will:
- Read videos from the source folder
- Sample k frames evenly from each video
- Save the frames as PNG files in a subfolder named after the video

### 2. Model Evaluation (`evaluate_videos.py`)

This script evaluates the sampled frames using various vision-language models on Azure platform against physics-based standards. It supports multiple models including GPT-4o and other compatible models.

Note: Before running the evaluation, make sure you have run `sample_video_frames.py` first. The evaluation script will only process videos that have corresponding folders in the `videos/sampled_frames` directory. For example, if you have videos `001-1.mp4`, `001-2.mp4`, and `001-3.mp4`, but only ran the sampling script on `001-1.mp4`, then only `001-1` will be evaluated.

Usage:
```bash
# Basic usage with default parameters
python evaluate_videos.py

# Specify model and number of frames
python evaluate_videos.py --gpt_model gpt-4o --total_frames 8

# Run evaluations in parallel
python evaluate_videos.py --run_in_parallel

# Combine multiple options
python evaluate_videos.py --gpt_model gpt-4o --total_frames 8 --run_in_parallel
```

Command-line arguments:
- `--gpt_model`: Model to use for evaluation (default: "gpt-4o")
  - Supported models: "gpt-4o", "gpt-o1"
- `--total_frames`: Number of frames to evaluate per video (default: 8)
- `--run_in_parallel`: Run evaluations in parallel (default: False)

The script will:
- Read prompts and standards from `prompts-with-standard-and-index.json`
- Process each video's frames
- Generate evaluations using the specified Azure model
- Save results in JSON format

Additional parameters (configured in the script):
- `is_two_steps_prompt`: Whether to use two-step evaluation (default: False)
- `llm_prompt_type`: Type of prompt to use (default: "one_step")
  - Supported types: "one_step", "two_step_with_standard_first", "two_step_with_standard_last", "two_step_no_standard_first", "two_step_no_standard_last"

### 3. Results Analysis (`analyze_results.py`)

This script analyzes the evaluation results and calculates statistics.

Before running the analysis, make sure to:
1. Run the evaluation script (`evaluate_videos.py`) first
2. Check the output JSON files in `automatic_results/[model_name]/frame-[total_frames]/is_two_step_False/` to ensure they follow the expected format:
```json
{
    "data": {
        "Prompt": "...",
        "Physics": "...",
        "Basic_Standards": {
            "Objects": [...],
            "Event": "..."
        },
        "Key_Standards": [...],
        "Prompt_index": "..."
    },
    "model_name": "[model_name]",
    "response": {
        "Objects": "Yes/No",
        "Event": "Yes/No",
        "Standard_1": "Yes/No",
        "Standard_2": "Yes/No"
    }
}
```

Usage:
```bash
# Basic usage with default parameters
python analyze_results.py

# Specify model and number of frames
python analyze_results.py --gpt_model gpt-4o --total_frames 8
```

Command-line arguments:
- `--gpt_model`: Model to analyze results for (default: "gpt-4o")
  - Supported models: "gpt-4o", "gpt-o1"
- `--total_frames`: Number of frames to analyze (default: 8)

The script will analyze the results and calculate:
- Percentage of videos where Objects and Events are both "Yes"
- Percentage of videos where all Standards are "Yes"
- Percentage of videos where everything (Objects, Events, and Standards) is "Yes"

## Directory Structure

```
.
├── videos/           # Contains 6 example videos and sampled frames
│   ├── 001-1.mp4           # Example video for prompt 001-1
│   ├── 001-2.mp4           # Example video for prompt 001-2
│   └── sampled_frames/     # Generated by sample_video_frames.py
│       ├── 001-1/
│       ├── 001-2/
│       └── ...
├── automatic_results/
│   └── [model_name]/
│       └── frame-[total_frames]/
│           └── is_two_step_[is_two_steps_prompt]/
│               └── *.json
├── sample_video_frames.py
├── evaluate_videos.py
├── analyze_results.py
├── prompts-with-standard-and-index.json  # Contains evaluation standards
└── prompt_index_to_prompt.json          # Contains prompts for video generation
```
<!-- 
## Output Format

The evaluation results are saved in JSON files with the following structure:
```json
{
    "data": {
        "Prompt": "...",
        "Physics": "...",
        "Basic_Standards": {
            "Objects": [...],
            "Event": "..."
        },
        "Key_Standards": [...],
        "Prompt_index": "..."
    },
    "model_name": "[model_name]",
    "response": {
        "Objects": "Yes/No",
        "Event": "Yes/No",
        "Standard_1": "Yes/No",
        "Standard_2": "Yes/No"
    }
}
``` -->

## Analysis Results

The analysis script provides:
1. Percentage of videos where Objects and Events are both "Yes"
2. Percentage of videos where all Standards are "Yes"
3. Percentage of videos where everything (Objects, Events, and Standards) is "Yes"

These metrics help evaluate how well the AI-generated videos adhere to physics-based standards and contain the required objects and events. 


## 📖 Citation

If you find our work useful, please cite:

```bibtex
@article{gu2025phyworldbench,
  title     = {PhyWorldBench: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models},
  author    = {Gu, Jing and Liu, Xian and Zeng, Yu and Nagarajan, Ashwin and Zhu, Fangriu and Hong, Daniel and Fan, Yue and Yan, Qianqi and Zhou, Kaiwen and Liu, Ming-Yu and Wang, Xin Eric},
  journal   = {arXiv preprint arXiv:2507.13428},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.13428}
}

```
