# DynamicBind

## Conda 环境

### 方案1 （最简单）

使用提供的conda 环境

```bash
mkdir ./conda_dynamic_bind
tar -xvf dynamic_bind.tar.gz -C ./conda_dynamic_bind
conda activate ./conda_dynamic_bind
conda-unpack
```

### 方案2

DynamicBind模型基于PyTorch和PyTorch Geometric，conda环境可以这样安装： 

```bash
conda env create -f environment.yml
```

### 方案3

本环境基于Python 3.9.19和cuda12.1，其他版本的Python可能会导致某些包无法安装（兼容性问题）。如果您的系统cuda版本不一样，可能需要自行安装对应版本的Pytorch和PyTorch Geometric，否则可能会失去GPU支持。

```
pytorch=2.3.1
torch-cluster==1.6.3+pt23cu121
torch-geometric==2.5.3
torch-scatter==2.1.2+pt23cu121
torch-sparse==0.6.18+pt23cu121
torch-spline-conv==1.2.2+pt23cu121
```

## 数据

相关数据都在`data.tar.gz`中，包含处理好的动态和静态数据。可以使用

```bash
tar xaf data.tar.gz
```

进行解压。

## 模型测试

提供的训练好的模型如下：

```yaml
# CASF-2016
results/casf_no_md: CASF 没有MD数据训练
results/casf_md3000: CASF 用全部MD数据训练
results_casf_ablation_study/casf_md1000: CASF用1000条轨迹训练
results_casf_ablation_study/casf_md2000: CASF用2000条轨迹训练
results_casf_ablation_study/casf_md3000_10ns: CASF用10ns训练
results_casf_ablation_study/casf_md3000_100ns: CASF用100ns训练

# LP-PDBBind
results/lppdbbind_no_md: LPPDBBind没有MD数据训练
results/lppdbbind_md3000: LPPDBBind用全部MD数据训练
results_lppdbbind_ablation_study/lppdbbind_md300: LPPDBBind用300条轨迹训练
results_lppdbbind_ablation_study/lppdbbind_md1000: LPPDBBind用1000条轨迹训练
results_lppdbbind_ablation_study/lppdbbind_md3000_1ns: LPPDBBind用1ns训练
results_lppdbbind_ablation_study/lppdbbind_md3000_10ns: LPPDBBind用10ns训练
results_lppdbbind_ablation_study/lppdbbind_md3000_100ns: LPPDBBind用100ns训练
results_lppdbbind_ablation_study/lppdbbind_md3000_250ns: LPPDBBind用250ns训练
```

模型测试：
```bash
python inference.py results/casf_no_md
```

模型测试结果会存储在`test_scores`子文件夹里面。

## 模型训练

模型训练需要一个训练config文件：

```bash
python train.py configs/casf_md3000.yml
```

