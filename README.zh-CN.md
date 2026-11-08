# 长壁开采支护评估多模块代理框架

当前源码版本为 **2.0.0**，包括支护需求、位移场和正应力场三个独立模块。完整数据与模型ZIP已发布于Zenodo，版本DOI为[10.5281/zenodo.24049496](https://doi.org/10.5281/zenodo.24049496)，文件与本地已核验ZIP一致。

| 模块 | 方法 | 输出 |
|---|---|---|
| 支护需求 | TabPFN教师锚定CMCRNet | 搜索范围内未达标概率、有限需求与解码压力档位 |
| 位移场 | PCA与双成员MLP，第5步实施Guard64系数范围约束 | 水平位移、竖向位移，单位m |
| 正应力场 | PCA与树集成，各成员重建后按开采步加权 | 三个正应力分量，单位MPa，压缩为负 |

GitHub提供源码、示例和说明。完整960组支护开发数据、64组归档评价、2970张历史场快照、265张Guard64参考及50张计划新工况快照，连同25个冻结模型文件，放在约3.05 GB的独立ZIP中。

```bash
python -m pip install -r requirements.txt
python scripts/prepare_archive.py --archive /path/to/longwall-surrogate-v2.0.0_20261108.zip
python verify_files.py
python predict_support.py --dataset archived --output outputs/support.csv
python predict_fields.py --input examples/field_inputs_10.npz --output outputs/fields.npz
python evaluate.py --output-dir outputs/evaluation
```

完整ZIP文件名、SHA256与发布状态见[归档信息](metadata/archive.json)。如已解压，可将环境变量`LONGWALL_SURROGATE_ARTIFACTS`设置为含有`data`、`models`和`SHA256SUMS.txt`的归档目录。现有Zenodo DOI `10.5281/zenodo.20839925`标识旧版v1.0.0，当前三模块归档的版本DOI为`10.5281/zenodo.24049496`。

原有旧版材料已保存在`legacy/v1.0.0/`，旧标签与Release保留。当前接口仅输出两个位移与三个正应力分量。场模型采用数据驱动的系数学习，不含旧稿的力学残差训练路径。

支护复算使用归档教师预测缓存。任意新组合需额外提供兼容的教师锚定预测，第三方基础模型权重通过原渠道获取。原研究的质量状态与验收结果保持原状，复算软件通过与研究验收通过具有不同含义。详细说明见[英文首页](README.md)和[复现说明](docs/reproducibility.md)。
