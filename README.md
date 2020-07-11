# 基于遗传算法的抗攻击网络拓扑结构优化程序
* 程序文件介绍
    * train.py 迭代训练
    * GA.py 遗传算法
    * edge_distribution.py 边分布可视化
    * iter_random_attack.py 算法迭代结果可视化，随机攻击仿真及其可视化
    * targeted_node_attack.py 恶意节点攻击仿真及其可视化
    * targeted_edge_attack.py 恶意边攻击仿真及其可视化
* 程序运行
    * 需要安装的python模块
        * networkx
        * collections
        * operator
        * ...
    * 运行方式
        * 在安装了上述的python模块之后，运行train.py
        * 在完成了train.py 之后，可以分别单独运行其他的可视化和仿真的程序文件
* 汇报：基于遗传算法的抗攻击网络拓扑结构优化