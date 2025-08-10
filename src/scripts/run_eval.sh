
#### 2. 如何在命令行/终端中正确运行

在修改完 `.sh` 脚本后，您就可以像之前一样运行它们了。如果您想手动在命令行中执行，请确保您**当前位于项目的根目录** `multi_uav_pf_lstm_matd3\`，然后执行以下命令：

**对于训练：**
```bash
# Windows (CMD or PowerShell)
python -m src.train.trainer --config_default configs/default.yaml --config_exp configs/experiment1.yaml

# Linux / macOS / Git Bash on Windows
./scripts/run_train.sh