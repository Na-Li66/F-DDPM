torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond

torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz

torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=4 --batch=64 --algorithm_name="F-DDIM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDPM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz


torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond

torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz

torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=6 --batch=64 --algorithm_name="F-DDIM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=9 --batch=64 --algorithm_name="F-DDPM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz









torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond

torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz

torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=8 --batch=64 --algorithm_name="F-DDIM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=12 --batch=64 --algorithm_name="F-DDPM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz






torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond

torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond
torchrun --standalone --nproc_per_node=1 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s
torchrun --standalone --nproc_per_node=1 main.py --subdirs img64s --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img64s

torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs ffhq --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz

torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DDIM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz
torchrun --standalone --nproc_per_node=1 main.py --subdirs bedroom --seeds=0-49999 --NFE=15 --batch=64 --algorithm_name="F-DDPM" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz