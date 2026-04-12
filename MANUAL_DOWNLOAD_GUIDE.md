# Rice3k 数据集手动下载指南

## 🎯 目标数据集
- **数据集名称**: Rice3k Genomic Selection Dataset
- **来源**: RenqiChen/Genomic-Selection GitHub 项目
- **Google Drive**: https://drive.google.com/drive/folders/1H6XL9IHDvXR8Suq64bd1NxH_YghGdUYC

## 📂 数据结构
根据下载过程，该文件夹包含：
```
Rice3k/
├── Genotypic/           # 基因型数据
│   ├── B001.txt
│   ├── B002.txt
│   ├── ...
│   └── B054.txt
├── Phenotypic/          # 表型数据 (5个fold)
│   ├── Fold1/
│   ├── Fold2/
│   ├── Fold3/
│   ├── Fold4/
│   └── Fold5/
└── 其他文件...
```

## 🔧 手动下载步骤

### 步骤 1: 访问 Google Drive
1. 打开浏览器，访问: https://drive.google.com/drive/folders/1H6XL9IHDvXR8Suq64bd1NxH_YghGdUYC
2. 登录您的 Google 账号

### 步骤 2: 下载基因型数据
1. 进入 `Genotypic` 文件夹
2. 选择所有 B*.txt 文件 (B001.txt 到 B054.txt)
3. 右键选择 "下载"
4. **重要**: 如果文件太多，可以分批下载

### 步骤 3: 下载表型数据
1. 返回主文件夹
2. 进入 `Phenotypic` 文件夹
3. 下载所有 5 个 fold 的数据
4. 每个fold包含不同性状的表型数据

### 步骤 4: 文件整理
下载完成后，请按以下结构整理文件：

```
e:/GWAS/data/raw/rice3k/
├── genotype/
│   ├── B001.txt
│   ├── B002.txt
│   ├── ...
│   └── B054.txt
└── phenotype/
    ├── fold1/
    ├── fold2/
    ├── fold3/
    ├── fold4/
    └── fold5/
```

## 🔄 替代方案

### 方案 A: 使用 Google Drive 桌面应用
1. 安装 Google Drive Desktop
2. 将文件夹同步到本地
3. 复制到项目目录

### 方案 B: 分批下载
如果一次性下载失败：
1. 每次下载 5-10 个文件
2. 使用下载管理器 (如 IDM, Free Download Manager)
3. 确保网络稳定

### 方案 C: 联系作者
如果仍有问题，可以：
1. 访问 GitHub 项目: https://github.com/RenqiChen/Genomic-Selection
2. 通过 Issues 联系作者
3. 说明研究用途，请求直接分享

## 📊 数据格式说明

### 基因型数据格式 (B*.txt)
- 每行代表一个样本
- 每列代表一个 SNP
- 数值范围: 0, 1, 2 (基因型编码)

### 表型数据格式
- CSV 格式
- 包含样本 ID 和表型值
- 按 5-fold 交叉验证组织

## ⚠️ 注意事项

1. **文件大小**: 整个数据集可能较大 (几GB)
2. **下载时间**: 根据网络速度，可能需要 10-30 分钟
3. **存储空间**: 确保有足够的磁盘空间
4. **数据完整性**: 下载后检查文件数量和大小

## 🚀 下载后处理

下载完成后，运行：
```bash
# 处理 Rice3k 数据
python scripts/process_rice3k.py
```

## 🆘 故障排除

### 问题 1: 下载中断
- **解决方案**: 使用下载管理器支持断点续传

### 问题 2: 权限不足
- **解决方案**: 联系文件夹所有者申请访问权限

### 问题 3: 文件损坏
- **解决方案**: 重新下载损坏的文件

## 📞 获取帮助

如果遇到问题：
1. 查看项目 GitHub Issues
2. 联系数据集作者
3. 寻求技术支持

---

**最后更新**: 2026年3月25日
**状态**: 等待手动下载
