# 圣遗物刷取概率

基于 Python、Tkinter 和 NumPy 的原神圣遗物刷取概率计算器。

## 运行

```sh
pip install -r requirements.txt
python genshin_artifact_probability.py
```

填写 N（刷取数量）、v（总有效词条严格超过的阈值）、f（目标概率）中的两个，另一项留空后点击计算。反求 v 返回达到目标概率的最大整数阈值，反求 N 返回搜索范围内的最小刷取数量。

模型使用超几何分布和强化二项分布，通过五维联合累积分布有限差分计算各位置最优装备的联合分布；成功要求五个位置全部拥有合格装备。主词条概率及模型假设见问题描述.txt。结果取决于所列模型假设，并非游戏官方概率承诺。

## Windows 打包

```sh
pip install pyinstaller
python -m PyInstaller --noconfirm --onefile --windowed --name ArtifactCalculator genshin_artifact_probability.py
```

可执行文件位于 dist/ArtifactCalculator.exe。

计算结果.txt 包含默认主词条配置的结果，主词条组合比较.csv 包含不同组合在多个阈值下的概率。

## 许可

采用 MIT 许可证。
