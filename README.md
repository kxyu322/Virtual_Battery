# 一、项目简介
- 把“虚拟储能（Virtual Battery）”和“空调集群（TCLs）”联合起来，跟踪电网给出的 AGC（自动发电控制）信号。
- 电池响应快但容量有限，空调响应慢但容量大，两者互补，用来跟踪 AGC 曲线。
- 记录并输出：
	- 电池充放电功率及 SOC（荷电状态）
    - 各台空调的温度和开关状态
    - 空调集群功率
# 二、依赖环境
- Python 3.7+
- 推荐虚拟环境（venv、conda 等）
- 主要库：
    pip install numpy pandas matplotlib tqdm
- 如果后续有新增库，请同步更新 `requirements.txt`
# 三、仓库结构
.  
├── main.ipynb          # 核心仿真代码（Notebook 版）   
├── data/                    # 放 AGC 数据  
│   └── rto-regulation-signal-08-2024.xlsx
├── .attachments      # ipynb的附件  
├── requirements.txt    # 依赖列表  
└── README.md           # 本说明文档

# 五、与其他部分的交互
## 输入

| 参数         | 变量     | 单位     | 说明                        |
| ---------- | ------ | ------ | ------------------------- |
| 调节信号       | AGC    | kW     | 希望虚拟电池消纳或出力的功率值。时序变量      |
| 空调额定功率     | Pm     | kW     | 空调工作时消耗的功率                |
| 空调设定温度     | Tr     | Celsus | /                         |
| 空调设定温度波动范围 | Tdelta | Celsus | 希望空调将房间温度控制在Tr+-Tdelta范围内 |
| 外部空气温度     | Ta     | Celsus | Ta越高，空调消耗平均功率越高           |
## 输出

见`main.ipynb`中的`四、需要展示的数据`部分。具体画图的风格、展示的层级由zxh来确定。


