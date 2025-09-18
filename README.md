# mujoco
用于mujoco仿真代码的迭代

1. 统一现实与仿真的质量分布
2. 首先通过simulate_main文件对一个运动进行仿真
3. 得到的数据可以通过txt-csv-mat的方式在matlab中进行绘制
4. 目前已经有G1、C3、C4、Jphif、Jphir五个文件的辨识实验数据（格式为mat，目前能转成txt文件，但是输出效果不太好，后面可以考虑统一使用matlab绘制）
5. mujoco中的辨识实验可以考虑，循环仿真-保存数据-txt转mat文件-matlab中拟合