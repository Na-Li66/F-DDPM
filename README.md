## Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation<br>

**Abstract**

Stochastic diffusion samplers often require substantially more neural function evaluations (NFEs) than deterministic samplers to achieve comparable sample quality.[^preliminary] Although higher-order discretization accelerates deterministic sampling, its benefits for stochastic sampling remains unclear. We establish matching upper and worst-case lower bounds showing that stochastic samplers with higher-order approximations remain first-order convergent, even with exact score functions. Thus, increasing the approximation order does not generally improve the convergence order. Motivated by this limitation, we introduce forward-value discretization, a training-free framework that evaluates the data/noise prediction model at a lookahead estimate constructed from previous predictions. For deterministic sampling, the resulting *Forward DPMSolver* (*F-DPMSolver*), improves sample quality at low NFE budgets through its DDIM-based variant, despite remaining first-order convergent. For stochastic sampling, we develop *Forward DDPM* (*F-DDPM*), which evaluates the model at a cleaner lookahead point adjusted for subsequent noise injection. This estimate reuses both the previous model prediction and the previously injected noise. Both variants admit first-order convergence guarantees with one model evaluation per iteration. Across CIFAR-10, ImageNet, FFHQ, and LSUN, F-DDPM consistently achieves lower FID than the evaluated stochastic baselines at matched NFE budgets. With 1.5 times as many NFEs as deterministic baselines, it achieves competitive sample quality, narrowing the practical efficiency gap between stochastic and deterministic sampling.

[^preliminary]: Partial preliminary results of this work appeared in International Conference on Machine Learning 2026 ([paper](https://arxiv.org/abs/2512.24927)).

## Requirements

Please refer to [`environment.yaml`](./environment.yaml) for the complete list of dependencies.

You can create and activate the environment using Miniconda:

```bash
conda env create -f environment.yaml -n F-DDPM
conda activate F-DDPM
```

## Getting started

To reproduce the main results from our paper, simply run:

```bash
sh run.sh
```

## Pre-trained Models
We evaluate our method on four benchmark datasets using publicly available pre-trained diffusion models:
1. CIFAR-10 with EDM (Repository: https://github.com/NVlabs/edm)
2. ImageNet 64×64 with EDM2 (size S and L) (Repository: https://github.com/NVlabs/edm2)
3. ImageNet 512×512 with EDM2 (size XS and XXL) (Repository: https://github.com/NVlabs/edm2)
4. LSUN and FFHQ with Latent Diffusion (Repository: https://github.com/CompVis/latent-diffusion)

## Sample
To generate samples using a specified model and sampler, run:

```.bash
# Generate 1024 images using 4 GPUs
torchrun --standalone --nproc_per_node=4 main.py \
  --subdirs {images_dir} \
  --seeds=0-49999 \
  --NFE {NFE} \
  --batch {batch_size} \
  --algorithm_name "F-DPMSolver" \
  --model_name {model_name} \
  --order {order}
```

### Supported Models (model_name)
1. CIFAR10-cond
2. CIFAR10-uncond
3. ImageNet64-L
4. ImageNet64-S
5. ImageNet512-XXL
6. ImageNet512-XS
7. LSUN
8. FFHQ

### Baseline Samplers
You can also reproduce baseline results by replacing algorithm_name with:
1. F-DDPM
2. F-DDIM
3. DDIM
4. DPMSolver
5. UniPC

## Calculating FID

To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `./fid/fid.py` for edm and `./fid/calculate_metrics_func.py`:

```.bash
# Calculate FID with images_dir of CIFAR10
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --subdirs {images_dir}

# Calculate FID with images_dir of ImageNet
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img64.pkl --subdirs {images_dir}
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl --subdirs {images_dir}
```

Since latent diffusion does not provide official FID reference statistics, we preprocess the datasets and compute reference files manually. The reference statistics we used could be found in https://drive.google.com/drive/folders/1x1pEbZbD73giIOXUqLA3-h8W-GDORUym?usp=share_link.

The generated references are stored in the refs/ directory.

```.bash
# Image process and get reference of LSUN
unzip bedroom_train_lmdb
python ./fid/data_process/extract_data.py --lmdb bedroom_train_lmdb --out data/lsun/bedroom/1024 --fmt png --workers 8 --skip-existing
python ./fid/data_process/lsun_preprocess.py
python ./fid/fid.py ref --data=images_raw/bedroom/256 --dest=refs/lsun-bedroom.npz

# Image process and get reference of FFHQ
python ./fid/data_process/FFHQ_preprocess.py
python ./fid/fid.py ref --data= data/ffhq/256 --dest=refs/ffhq-256.npz
```

After generating the reference statistics, FID is calculated by
```.bash
# Calculate FID with images_dir of LSUN
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs {images_dir} --ref_path refs/lsun-bedroom.npz

# Calculate FID with images_dir of FFHQ
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs {images_dir} --ref_path refs/ffhq-256.npz
```

## Main Results

The following FID results are obtained by `run.sh`.

### F-DDPM (stochastic)

Settings: `--algorithm_name="F-DDPM"`, with `--NFE` in `{6, 9, 12, 15}`.

| NFE | CIFAR10-cond | CIFAR10-uncond | ImageNet64-S | ImageNet64-L | ImageNet512-XS | ImageNet512-XXL | LSUN | FFHQ |
| ---: | -----------: | -------------: | -----------: | -----------: | -------------: | --------------: | ---: | ---: |
| 6    | 15.11 | 21.85 | 24.01 | 25.63 | 37.86 | 36.71 | 15.09 | 22.57 |
| 9    | 6.85  | 8.63  | 8.63  | 8.90  | 10.86 | 9.87  | 5.17  | 9.04  |
| 12   | 4.13  | 4.62  | 4.38  | 4.30  | 5.75  | 4.18  | 4.50  | 7.33  |
| 15   | 3.23  | 3.34  | 3.06  | 3.06  | 4.47  | 2.92  | 4.47  | 7.31  |

### F-DDIM (deterministic)

Settings: `--algorithm_name="F-DDIM"`, with `--NFE` in `{4, 6, 8, 10}`.

| NFE | CIFAR10-cond | CIFAR10-uncond | ImageNet64-S | ImageNet64-L | ImageNet512-XS | ImageNet512-XXL | LSUN | FFHQ |
| ---: | -----------: | -------------: | -----------: | -----------: | -------------: | --------------: | ---: | ---: |
| 4    | 18.44 | 25.02 | 20.55 | 22.35 | 50.96 | 52.32 | 12.17 | 15.78 |
| 6    | 7.57  | 9.47  | 6.67  | 7.20  | 14.33 | 14.04 | 5.56  | 7.89  |
| 8    | 4.32  | 4.92  | 3.50  | 3.64  | 6.18  | 5.04  | 4.92  | 6.80  |
| 10   | 3.18  | 3.46  | 2.56  | 2.51  | 4.57  | 3.06  | 4.65  | 6.46  |

The paper pairs F-DDIM/F-DDPM budgets as 4/6, 6/9, 8/12, and 10/15, so F-DDPM uses **1.5 times** as many evaluations in each comparison.
