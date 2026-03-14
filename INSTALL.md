# Installation & Setup Guide

Quick setup guide for NVFlow - a lightweight orchestration tool for Slurm clusters.

## 📋 Steps

1. [Prerequisites](#prerequisites)
2. [Setup Containers](#setup-containers)
3. [Download Models](#download-models)
4. [Configure Your Cluster](#configure-your-cluster)
5. [Verify Installation](#verify-installation)

---

## Prerequisites

> **Note:** This guide assumes you've already completed the [README.md](README.md) setup (installed `uv`, cloned the repo, ran `uv sync`).

### Cluster Setup Requirements

**Required:**
- **Slurm cluster** access with SSH keys

**Only needed for converting Docker images to .sqsh format:**
- **yq** - YAML parser ([install guide](https://github.com/mikefarah/yq))
- **curl** - for downloading configs
- **enroot** - on cluster nodes (for container conversion)

> **Note:** If you already have `.sqsh` container images, skip to [Configure Your Cluster](#configure-your-cluster).

**yq (YAML parser):**
```bash
# Check if installed
yq --version

# If not installed:
# macOS
brew install yq

# Linux
mkdir -p $HOME/bin
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq
chmod +x $HOME/bin/yq

# Add to PATH (if $HOME/bin not already in PATH)
echo 'export PATH="$HOME/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc
```

**curl & enroot:**
```bash
curl --version     # Usually pre-installed
enroot version     # Run on cluster node
```

**Get your cluster info:**
- Slurm account: `sacctmgr show associations user=$USER` (look for the Account column)
- Available partitions: `sinfo`
- Storage paths for data/models/containers

---

## Setup Containers

NeMo-Skills requires Docker containers converted to `.sqsh` format for running on Slurm clusters.

**Required containers (4):**

| Container | Source | Action |
|-----------|--------|--------|
| `nemo-skills` | NeMo-Skills Dockerfiles | **Build** (see Step 1) |
| `vllm` | NeMo-Skills Dockerfiles or `vllm/vllm-openai` | **Build** or pull from Docker Hub |
| `sglang` | `lmsysorg/sglang` | Pull from Docker Hub |
| `nemo-rl` | NeMo-Skills Dockerfiles | **Build** (see Step 1) |

**Optional containers** (not currently used by any NVFlow recipes):

| Container | Source | Action |
|-----------|--------|--------|
| `megatron` | NeMo-Skills Dockerfiles | Build |
| `sandbox` | NeMo-Skills Dockerfiles | Build |
| `verl` | NeMo-Skills Dockerfiles | Build |
| `trtllm` | `nvcr.io/nvidia/tensorrt-llm/release` | Pull from NGC |

### Step 1: Build Docker Images

Clone the NeMo-Skills repo at the **exact commit pinned by NVFlow** to ensure compatibility. The pinned commit is defined in [`pyproject.toml`](pyproject.toml):

```bash
# Clone NeMo-Skills and check out the pinned commit
git clone https://github.com/NVIDIA/NeMo-Skills.git
cd NeMo-Skills
git checkout 7d6c49a51efb441b61db3e78f6ffa2f04c9a68ef
```

> **Tip:** Always use the commit hash from `pyproject.toml` (search for `nemo-skills @`). Building from a different version may cause incompatibilities.

Build the required images using the [NeMo-Skills Dockerfiles](https://github.com/NVIDIA/NeMo-Skills/tree/7d6c49a51efb441b61db3e78f6ffa2f04c9a68ef/dockerfiles):

```bash
# Build the required containers
./dockerfiles/build.sh dockerfiles/Dockerfile.nemo-skills
./dockerfiles/build.sh dockerfiles/Dockerfile.vllm
./dockerfiles/build.sh dockerfiles/Dockerfile.nemo-rl

# Or build directly with docker
docker build -t nemo-skills:latest -f dockerfiles/Dockerfile.nemo-skills .
docker build -t nemo-skills-vllm:latest -f dockerfiles/Dockerfile.vllm .
docker build -t nemo-skills-nemo-rl:latest -f dockerfiles/Dockerfile.nemo-rl .
```

> **Note:** For `vllm`, you can alternatively pull a pre-built image directly from Docker Hub (`vllm/vllm-openai`) instead of building from the Dockerfile.

For optional containers (`megatron`, `sandbox`, `verl`), build them the same way using their respective Dockerfiles. For arm64 builds, see the [multi-platform instructions](https://github.com/NVIDIA/NeMo-Skills/tree/7d6c49a51efb441b61db3e78f6ffa2f04c9a68ef/dockerfiles#building-for-arm64aarch64).

### Step 2: Push Images to a Registry

After building, push the images to a container registry accessible from your cluster (Docker Hub, NGC, or a private registry):

```bash
# Tag and push the images you built
docker tag nemo-skills:latest your-registry/nemo-skills:latest
docker push your-registry/nemo-skills:latest

docker tag nemo-skills-vllm:latest your-registry/nemo-skills-vllm:latest
docker push your-registry/nemo-skills-vllm:latest

docker tag nemo-skills-nemo-rl:latest your-registry/nemo-skills-nemo-rl:latest
docker push your-registry/nemo-skills-nemo-rl:latest

# Repeat for any optional images you built (e.g., megatron, sandbox, verl)
```

> **Why push?** Slurm cluster nodes typically don't have Docker installed, so `enroot` needs to pull images from a registry. Pushing to a registry also lets the automated setup script work.

### Step 3: Update `containers.yaml`

Edit [`cluster_configs/containers.yaml`](cluster_configs/containers.yaml) to update image references with your registry paths:

```yaml
containers:
  # Required - pull from official registry (no changes needed)
  sglang: lmsysorg/sglang:v0.5.4

  # Required - replace with your own built images
  nemo-skills: your-registry/nemo-skills:latest
  vllm: your-registry/nemo-skills-vllm:latest
  nemo-rl: your-registry/nemo-skills-nemo-rl:latest

  # Optional
  # megatron: your-registry/nemo-skills-megatron:latest
  # sandbox: your-registry/nemo-skills-sandbox:latest
  # verl: your-registry/nemo-skills-verl:latest
```

### Step 4: Convert to .sqsh Format

Choose one of the following methods to convert your container images to `.sqsh` format for Slurm.

#### Option A: Automated Setup (Recommended)

Use the setup script to download from your registry and convert all containers in parallel:

```bash
sbatch --account=YOUR_ACCOUNT scripts/setup_containers.sh ./containers
```

The script reads image references from `cluster_configs/containers.yaml`, pulls them via `enroot`, and converts to `.sqsh` format. See [the script](scripts/setup_containers.sh) for options (`--platform`, `--force`).

**Check progress:**
```bash
tail -f outputs/logs/slurm-containers-<jobid>.out
```

#### Option B: Manual Conversion

Convert images one at a time using `enroot` on a cluster node:

```bash
# Import from your registry
enroot import docker://your-registry/nemo-skills:latest
enroot import docker://your-registry/nemo-skills-vllm:latest
enroot import docker://your-registry/nemo-skills-nemo-rl:latest

# Import from official registries (for sglang, etc.)
enroot import docker://lmsysorg/sglang:v0.5.4
```

Move the resulting `.sqsh` files to your cluster's container storage path.

---

## Download Models

> ⚠️ **Important:** Pre-download models to your cluster storage before running workflows.
>
> **Why this matters:**
> - Avoids wasting expensive GPU time on downloads
> - Prevents race conditions when multiple jobs start simultaneously
> - Large models (10-100+ GB) can take hours to download
> - Network failures during jobs cause workflow failures

### Using hf download

**Note:** `hf` CLI is included with nemo-skills (via `huggingface-hub`).

```bash
# Download model to your cluster storage
uv run hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /path/to/models/hf_models/Qwen/Qwen3-4B-Instruct-2507

# Example:
uv run hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /lustre/fs1/.../models/hf_models/Qwen/Qwen3-4B-Instruct-2507
```

**Storage location:** Models should go in your mounted HuggingFace models directory (see cluster config `mounts` section).

### Mount Path in Cluster Config

Ensure your cluster config has the models directory mounted (cluster config creation is explained in the [Configure Your Cluster](#configure-your-cluster) section below):

```yaml
mounts:
  - /cluster/path/to/models/hf_models:/hf_models  # Maps to /hf_models inside containers
```

### Using in Workflows

Reference models using the **container mount path** (`/hf_models`):

```yaml
stage_kwargs:
  model: /hf_models/Qwen/Qwen3-4B-Instruct-2507  # Path inside container
  server_type: sglang
```

**Tip:** Download commonly used models once and reuse across all workflows.

---

## Configure Your Cluster

### Step 1: Create Your Cluster Config

```bash
# Copy template
cp cluster_configs/template-slurm.yaml cluster_configs/my_cluster.yaml
```

### Step 2: Edit Your Config

Edit `cluster_configs/my_cluster.yaml` and replace all `<PLACEHOLDER>` values:

1. **SSH settings** - Your cluster login node, username, SSH key path (ONLY for remote job submission from local machine)
2. **Slurm account/partition** - Run `sacctmgr show associations user=$USER` and `sinfo`
3. **Container paths** - Copy from `outputs/logs/slurm-containers-<jobid>.out` after running setup_containers.sh
4. **Mount points** - Map your cluster paths to container paths
5. **Environment variables** - Set `HF_HOME` and any API keys

> **Note:** The template includes detailed comments for each section. Your personal config (`my_cluster.yaml`) is gitignored to protect secrets.
>
> 📖 **For detailed documentation of all configuration fields, see the [Cluster Configuration Guide](docs/cluster-configuration.md)**.

---

## Verify Installation

```bash
# 1. Test NeMo-Skills import
uv run python -c "from nemo_skills.pipeline.cli import generate; print('✅ OK')"

# 2. Check containers exist
ls -lh <PATH_TO_CONTAINERS>/*.sqsh

# 3. Test SSH to cluster
ssh -i <PATH_TO_SSH_KEY> <YOUR_USERNAME>@<YOUR_CLUSTER_LOGIN_NODE> "echo '✅ SSH OK'"

# 4. Test cluster config loads
uv run python -c "from omegaconf import OmegaConf; OmegaConf.load('cluster_configs/my_cluster.yaml'); print('✅ Config OK')"

# 5. List available stages
uv run nflow list-stages
```

---

## Troubleshooting

### Python 3.12 not found
```bash
# UV can install it for you
uv python install 3.12
uv sync
```

### NeMo-Skills not found
```bash
uv sync --reinstall
```

### yq not found
```bash
# macOS
brew install yq

# Linux
mkdir -p $HOME/bin
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq
chmod +x $HOME/bin/yq

# Add to PATH if needed
echo 'export PATH="$HOME/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc
```

### enroot not available
```bash
# On cluster
module load enroot  # if available
# Or contact your cluster admin
```

### SSH connection failed
```bash
# Check key permissions
chmod 600 <PATH_TO_SSH_KEY>

# Test manual connection
ssh -i <PATH_TO_SSH_KEY> <YOUR_USERNAME>@<YOUR_CLUSTER_LOGIN_NODE>
```

### Container paths wrong
- Use **absolute paths** in cluster config
- Check file exists: `ls -l <PATH_TO_CONTAINERS>/<container>.sqsh`
- Re-run container setup if needed

### Slurm jobs won't submit
- Verify account: `sacctmgr show associations user=$USER`
- Check partition: `sinfo -p <YOUR_GPU_PARTITION>`
- Look at job logs in `ssh_tunnel.job_dir`

### Ray Cluster Initialization Hangs

**Problem:** Ray cluster hangs during initialization, workers fail to connect

**Symptoms:**
- Training jobs hang after "Starting Ray cluster"
- Error: `execve(): bad interpreter: No such file or directory`
- Ray workers show connection failures in logs

**Cause:** SLURM container reattachment issues (SLURM 25.x may be affected)

**Solution:**
Add `ray_template` to your cluster config:

```yaml
# In cluster_configs/my_cluster.yaml
executor: slurm
ray_template: "ray_enroot.sub.j2"  # Fixes Ray cluster initialization
```

**Check your SLURM version:**
```bash
scontrol show config | grep SLURM_VERSION
# Confirmed: SLURM 25.11.2 needs this fix, SLURM 24.x works without it
```

---

## Next Steps

✅ **Cluster setup complete!**

You can now run workflows on your cluster. Set the config directory:

```bash
export NEMO_SKILLS_CONFIG_DIR=/path/to/nvflow/cluster_configs
```

Then head back to the [README.md](README.md#-quick-start) Quick Start section to run your first workflow.

---

## Reference

- **NeMo-Skills**: https://github.com/NVIDIA/NeMo-Skills
- **NeMo-Skills Dockerfiles**: https://github.com/NVIDIA/NeMo-Skills/tree/main/dockerfiles
- **Official Container Config**: https://github.com/NVIDIA/NeMo-Skills/blob/main/cluster_configs/example-slurm.yaml
- **Slurm Docs**: https://slurm.schedmd.com/
- **Enroot**: https://github.com/NVIDIA/enroot

**Need help?** Check the [Cluster Configuration Guide](docs/cluster-configuration.md) or ask your team/cluster admin.
