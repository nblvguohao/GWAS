# GitHub与服务器打包与复现流程说明

## 1. GitHub上传包（精简版）
- 只包含代码、配置、README、部分小型processed数据（rice469等）、部分results示例
- 不含data/raw/和大体积数据
- 用于开源、同行评议、代码审查

### 步骤
1. 运行：
   bash scripts/pack_for_github.sh
2. 上传github_upload.zip到GitHub（或解压后git add/commit/push）
3. 他人可clone仓库，按README/QUICK_START.md复现小数据集流程

## 2. 服务器全量包
- 包含全部data/processed/和results/，便于完整复现
- 不含data/raw/（如需原始数据，按README下载）

### 步骤
1. 运行：
   bash scripts/pack_for_server.sh
2. 上传server_upload.zip到服务器
3. 服务器解压：
   unzip server_upload.zip
4. 按README/QUICK_START.md配置环境，运行主流程脚本

## 3. 复现建议
- 推荐先用小型数据（rice469等）测试流程
- 全量数据复现前，先检查依赖和环境
- 如遇大文件，建议用tar.gz打包

## 4. 注意事项
- .gitignore已配置，避免大文件误上传GitHub
- processed和results目录结构与CLAUDE.md方案一致
- 若需补充数据或结果，手动拷贝到对应目录后重新打包
