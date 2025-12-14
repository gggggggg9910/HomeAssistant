# 解决依赖冲突的方案

## 问题：pip 依赖冲突

当运行 `pip install 'datasets<3.0.0'` 时出现依赖冲突：

```
openxlab 0.1.3 requires requests~=2.28.2, but you have requests 2.32.5 which is incompatible.
openxlab 0.1.3 requires tqdm~=4.65.0, but you have tqdm 4.67.1 which is incompatible.
```

## 解决方案

### 方法1：强制降级（推荐）

```bash
# 强制降级 datasets 和相关依赖
pip install 'datasets<3.0.0' --force-reinstall --no-deps

# 或者更激进的方案
pip install 'datasets==2.20.0' --force-reinstall --no-deps
```

### 方法2：先卸载冲突包

```bash
# 卸载可能冲突的包
pip uninstall -y openxlab tqdm requests

# 然后安装兼容版本
pip install 'datasets<3.0.0'
pip install 'tqdm>=4.65.0,<4.66.0' 'requests>=2.28.0,<2.29.0'
```

### 方法3：使用 conda（如果适用）

```bash
# 如果使用 conda
conda install datasets=2.20.0
```

### 方法4：接受冲突并测试

依赖冲突不一定影响实际功能，可以先测试：

```bash
python -c "import modelscope; print('modelscope works')"
```

如果导入成功，程序应该能正常运行。

## 验证解决方案

安装后测试：

```bash
python -c "import datasets; print('datasets version:', datasets.__version__)"
python -c "import modelscope; print('modelscope imported successfully')"
```

## 如果仍有问题

如果上述方法都不行，可以：

1. **忽略冲突** - 很多情况下依赖冲突不会影响实际功能
2. **使用虚拟环境** - 创建新的虚拟环境重新安装
3. **降级 modelscope** - 使用旧版本的 modelscope

```bash
# 降级 modelscope 到更兼容的版本
pip install 'modelscope<1.15.0'
```

## 当前状态检查

运行以下命令检查当前状态：

```bash
# 检查版本
python -c "import datasets; print('datasets:', datasets.__version__)"
python -c "import modelscope; print('modelscope imported')"

# 检查是否有 get_metadata_patterns 函数
python -c "from datasets.data_files import get_metadata_patterns; print('Function exists')"
```