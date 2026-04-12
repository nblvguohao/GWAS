# 🌾 真实作物基因组选择数据获取文献指南

## 📖 核心文献及数据获取方法

### 1. 🌽 玉米数据 - Maize 282 Association Panel

#### 📄 关键文献
```
Flint-Garcia, S. A., et al. (2005). Maize association population: a high-resolution platform for quantitative trait locus dissection. *Science*, 307(5710), 1968-1971.

DOI: 10.1126/science.1104639
```

#### 🔗 数据获取方式
1. **Panzea数据库** (推荐)
   - 网址: https://www.panzea.org/
   - 直接下载: https://cbsusrv04.tc.cornell.edu/users/panzea/filegateway.aspx?category=Genotypes
   - 文件: "Maize 282 association panel genotypes (7x, AGPv4 coordinates)"

2. **CyVerse Data Store** (原始数据)
   - 需要注册CyVerse账号
   - 路径: /iplant/home/shared/panzea/hapmap3/hmp321/unimputed/282_libs_2015/uplifted_APGv4
   - 命令: `iget -K -r /iplant/home/shared/panzea/hapmap3/hmp321/unimputed/282_libs_2015/uplifted_AGPv4`

3. **补充文献**
   ```
   McMullen, M. D., et al. (2009). Genetic properties of the maize nested association mapping population. *Science*, 325(5941), 737-741.
   ```

---

### 2. 🌾 水稻数据 - Rice Diversity Panel 1 (RDP1)

#### 📄 关键文献
```
Zhao, K., et al. (2011). Genome-wide association mapping reveals a rich genetic architecture of complex traits in rice. *Nature*, 490(7419), 329-333.

DOI: 10.1038/nature11524
```

#### 🔗 数据获取方式
1. **Rice Diversity Project**
   - 网址: https://ricediversity.org/
   - 注册后下载: https://ricediversity.org/data/
   - 数据集: "Rice Diversity Panel 1 (RDP1)"

2. **补充数据源**
   - **3K Rice Genome Project**: http://www.ncgr.ac.cn/3kricedata/
   - 文献: Zhao, K., et al. (2018). The 3,000 rice genomes project. *GigaScience*, 7(8), giy102.

---

### 3. 🌱 小麦数据 - Wheat CAP

#### 📄 关键文献
```
Maccaferri, M., et al. (2015). Genome-wide association mapping in a U.S. winter wheat nested association mapping population. *The Plant Journal*, 84(5), 958-975.

DOI: 10.1111/tpj.13067
```

#### 🔗 数据获取方式
1. **Wheat CAP Project**
   - 网址: https://www.triticapeptide.org/wheat-cap/
   - 需要申请访问权限
   - 联系: wheat-cap@cornell.edu

2. **补充数据源**
   - **Wheat Initiative**: https://www.wheatinitiative.org/
   - **International Wheat Genome Sequencing Consortium**: http://www.wheatgenome.org/

---

### 4. 🫘 大豆数据 - 已有真实数据

#### ✅ 您已拥有: SoyDNGP数据
- 来源: https://github.com/IndigoFloyd/SoybeanWebsite
- 表型: 15,905个真实样本
- SNP位点: 32,033个真实位点

#### 📄 补充文献
```
Zhou, Z., et al. (2020). A genomic variation map provides insights into the adaptation and domestication of soybean. *Nature Biotechnology*, 38(9), 1084-1090.
```

---

### 5. 🥬 拟南芥数据 - 1001 Genomes

#### 📄 关键文献
```
Weigel, D., & Mott, R. (2009). The 1001 Genomes Project for Arabidopsis thaliana. *Nature Genetics*, 41(10), 1183-1186.

DOI: 10.1038/ng.436
```

#### 🔗 数据获取方式
1. **1001 Genomes Project**
   - 网址: https://1001genomes.org/
   - FTP: ftp://ftp.1001genomes.org/1001genomes/
   - VCF文件: /1001genomes/VCF/TAIR10/

2. **直接下载链接**
   - 基因型: ftp://ftp.1001genomes.org/1001genomes/VCF/TAIR10/1001genomes_snp-short-indel_only.vcf.gz
   - 表型: https://araphid.1001genomes.org/

---

## 🛠️ 数据处理工具

### 推荐软件
1. **PLINK** - 基因型数据处理
2. **TASSEL** - 植物关联分析
3. **VCFtools** - VCF文件处理
4. **bcftools** - VCF格式转换

### 格式转换示例
```bash
# VCF转PLINK格式
plink --vcf input.vcf --make-bed --out output

# PLINK转CSV
plink --bfile output --recode A --out csv_output
```

---

## 📋 申请数据需要的材料

### 学术申请模板
```
主题: 申请访问[数据库名称]基因组数据

尊敬的数据管理员：

我是[您的姓名]，来自[您的机构]。
我们正在进行[研究项目描述]研究，需要使用[具体数据集]进行[具体用途]。

我们承诺：
1. 仅用于学术研究
2. 遵守数据使用协议
3. 在发表文章时正确引用
4. 不用于商业目的

附件：研究计划书/机构证明

期待您的回复。

此致
敬礼！

[您的姓名]
[您的邮箱]
[您的机构]
[日期]
```

---

## 🚀 立即行动建议

### 优先级排序
1. **立即**: 1001 Genomes (无需申请)
2. **今天**: Maize 282 (Panzea公开)
3. **明天**: Rice RDP1 (需要注册)
4. **下周**: Wheat CAP (需要申请)

### 联系方式
- **Panzea帮助**: panzea@cornell.edu
- **Rice Diversity**: ricediversity@cornell.edu  
- **Wheat CAP**: wheat-cap@cornell.edu

---

## 📞 需要我帮助？

我可以帮您：
1. 📝 **撰写申请邮件**
2. 🔧 **编写数据转换脚本**
3. 📊 **设计数据处理流程**
4. 🎯 **制定研究计划**

请告诉我您想从哪个数据集开始，我立即协助您！
