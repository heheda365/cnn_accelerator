# 量化测试
w = 2, 量化到了三元(-1, 0, 1)
A = 4, 激活后用4位表示
## bn_act层
使用二分查找实现，只需要保存inc公差 和bias(其位宽根据具体情况做计算)
## 准确率
用C语言测试时准确率 9782/10000, 与用pytorch测试，相差1%左右的准确率损失，应该时软件编写存在bug
