时间序列异常检测实验工具脚本使用说明
======
说明
------
本工具用于进行时间序列异常检测模型实验，基于PyTorch框架实现的深度学习训练脚本。
为了方便实验创新，本工具封装了dataloader、训练流程、测试流程、评估流程、日志记录与模型导出功能，
并内置了部分数据集、预处理算法、模型与评估方法。
在进行创新实验时，可以仅修改数据集、预处理算法、模型与评估方法四个内容中的一个就开始快速实验，节省实验的冗余代码开发。
数据集、预处理算法、模型与评估方法四部分内容进行自定义时，请按照指定格式设计。
实验的参数请通过config.ini文件进行设置。

如需特殊接口，请联系开发人员郑延钦，邮箱:zhengyanqin@iie.ac.cn

安装
------
环境依赖

python==3.9.7

    * eventlet==0.33.3
    * gevent==22.10.2
    * loguru==0.6.0
    * numpy==1.21.2
    * pandas==1.4.1
    * scikit_learn==1.2.2
    * torch==1.10.2
    * tqdm==4.63.1


python第三方库安装方法

    pip3 install -r requirements.txt
    
配置文件参数说明
------
config.ini文件用于配置相关设置，包括数据集、算法模型、预处理方法、评估方法、模型参数加载、实验基本参数与自定义参数。

    [Dataset]
    # 填写数据集名称（文件夹名称），同时训练测试多个数据集，请用逗号隔开
    # 请确保数据集已经按照要求格式放置在./dataset路径下
    name=SWAT
    
    [Method]
    # 填写模型名称（文件名称，不用写".py"），同时训练测试多个算法，请用逗号隔开
    # 请确保模型文件已经按照要求格式放置在./model路径下
    # 模型名称请全部由字母组成
    name=MtadGatAtt
    
    [Preprocess]
    # 填写预处理方法名称（文件名称，不用写".py"）
    # 请确保预处理方法文件已经按照要求格式放置在./preprocess路径下
    name=standardization
    
    [Evaluation]
    # 填写评估方法名称（文件名称，不用写".py"），同时使用多个评估方法，请用逗号隔开
    # 请确保评估方法文件已经按照要求格式放置在./evaluation路径下
    name=f1,f1pa,ftad,affiliation
    
    [ModelPath]
    # 模型加载先前参数，请用"模型名_数据集名"作为变量名
    # 路径名称从TSAD文件夹下开始写相对路径，或者直接写绝对路径
    MtadGatAtt_SWAT=none
    
    [BaseParameters]
    # train表示是否训练模型，请填写true或者false，为false时，仅测试模型
    train=true
    # epoch表示训练轮数，请填写正整数
    epochs=20
    # batch-size表示每个batch的尺寸，请填写正整数
    batch_size=32
    # learning-rate表示学习率，建议不要高于0.001
    learning_rate=1e-4
    # device表示使用的设备，可以选auto、cpu、cuda，其中auto代表自动判断，有gpu使用gpu，没有则使用cpu；选gpu时需要选择gpu序号，例如cuda:0
    device=auto
    
    [CustomParameters]
    # 此处可以填入自定义参数，数量不限，用于填写特殊需求输入，例如模型文件所需的所有外部参数，构建数据集的部分参数
    # 输入数据长度input_size，即截取数据的窗口长度
    input_size=60
    # 输出数据长度output_size
    output_size=1
    # 截取数据的窗口的移动步长
    step=1
    # 数据集文件（csv）中，第一行是否需要忽略（第一行为列名的话需要忽略）
    del_column_name=true
    # 数据集文件（csv）中，第一列是否是时间戳
    time_index=true
    
启动与停止
------
通过运行main.py脚本文件，启动程序。可以选择 start | stop | daemon 三个参数。

* start是启动普通进程，前台运行，日志实时输出并本地记录，可输出训练进度条。
* stop是停止当前实验进程。
* daemon是启动守护进程，后台运行程序，日志仅保存本地，日志不包含训练进度条内容。

如无输入参数，默认选择start参数。使用方法如下所示：

    python3 main.py start/stop/daemon  
    
本工具会输出实验日志、所有模型的最佳检测标签、模型参数文件到./records路径下。
    
数据集文件配置说明
-----
所有数据集文件，均存放在./dataset路径下（git版无法上传）。
数据集可以包含子数据集，数据以csv格式文件存储，必须包含train.csv和test.csv。

* train.csv和test.csv的数据内容必须全为数值型数据，第一行可保留列名。
* train.csv和test.csv如包含时间戳，请置于第一列。
* test.csv最后一列为异常标签，建议用0表示正常数据，1表示异常数据（如标签含义相反，本工具会自动识别并反转标签）。

数据集路径示例如下：

    └── dataset
        ├── 数据集1名称
        │   ├── 子数据集1名称
        │   │   ├── train.csv
        │   │   ├── test.csv
        │   ├── 子数据集2名称
        ...
        │   └── 子数据集m名称
        ├── 数据集2名称
        │   ├── train.csv
        │   └── test.csv
        ...
        └── 数据集n名称
    
预处理文件配置说明
----
所有预处理方法的文件，均存放在./preprocess路径下，并写入__init__文件。
每个预处理方法单独建立一个文件，新建预处理方法的命名全部采用英文和数字组成，文件名称直接使用预处理方法的命名。

文件内需包含MyDataset类，MyDataset类继承torch.utils.data.Dataset，必须定义__len__、__getitem__两个方法。
各个方法的具体设计要求，详见./preprocess/template.py文件。

模型文件配置说明
-----
所有模型结构的文件，需存放于./method路径下，并写入__init__文件。
每个模型单独建立一个文件，新建模型的命名全部采用英文和数字组成，文件名称直接使用模型的命名。

文件内需包含Model类，Model类继承nn.Module，必须定义forward方法、loss方法和detection方法。
各个方法的具体设计要求，详见./method/template.py文件。

评估方法文件配置说明
-----
所有评估方法的文件，需存放于./evaluation路径下，并写入__init__文件。
每个评估方法单独建立一个文件，新建评估方法的命名全部采用英文和数字组成，文件名称直接使用模型的命名。

文件内需定义evaluate方法，方法的输入输出设计详见./evaluation/template.py文件。


