# 上传说明

本地整理后的项目目录为：

```text
/home/zhongjialin/projects/iree-optimization
```

## 在 GitHub 创建仓库

1. 打开 `PLC-CQU` 组织的仓库页面。
2. 点击 **New repository**。
3. 建议配置：
   - Owner: `PLC-CQU`
   - Repository name: `iree-optimization`
   - Visibility: `Private`
   - 不要勾选初始化 README、`.gitignore` 或 license，因为本地已经准备好了。
4. 点击 **Create repository**。

## 本地提交并上传

如果本机还没有配置 Git 身份，先执行一次：

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

使用 SSH：

```bash
cd /home/zhongjialin/projects/iree-optimization
git add -A
git commit -m "Add IREE optimization project"
git branch -M main
git remote add origin git@github.com:PLC-CQU/iree-optimization.git
git push -u origin main
```

使用 HTTPS：

```bash
cd /home/zhongjialin/projects/iree-optimization
git add -A
git commit -m "Add IREE optimization project"
git branch -M main
git remote add origin https://github.com/PLC-CQU/iree-optimization.git
git push -u origin main
```

如果已经配置过 `origin`，改地址即可：

```bash
git remote -v
git remote set-url origin git@github.com:PLC-CQU/iree-optimization.git
git push -u origin main
```

## 注意事项

不要把模型权重、ONNX、VMFB、IRPA、build 目录、日志和临时 tensor dump 加入 Git。这些内容已通过 `.gitignore` 排除，应该本地生成或放到外部 artifact 存储。
