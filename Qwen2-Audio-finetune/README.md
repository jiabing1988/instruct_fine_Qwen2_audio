# Qwen2-Audio-finetune  
This is a repository prepared for fine-tuning Qwen2-Audio, supporting both GPU and NPU. It supports reading and writing data in ark and wav formats, and is compatible with DDP, DeepSpeed, and LoRA.  

# Requirements  
The following are the dependencies required to run this project:  
```plaintext  
numpy==1.26.0  
torch==2.1.0  
torch-npu==2.1.0.post10 (if using NPU)  
torchaudio==2.1.0  
torchvision==0.16.0  
soundfile  
transformers==4.46.3  
hydra  
OmegaConf  
deepspeed  
```  

# Train  
## Data Preparation  
Please refer to the sample data path `/data/aishell-1` to prepare your data.  
```  
multitask.jsonl  
my_wav.scp  
multiprompt.jsonl  
```  
Modify the following two variables in `train.sh`, or simply use the default provided data without preparation:  
```  
TRAIN_DATA_PATH  
EVAL_DATA_PATH  
```  

## Configuration Preparation  
Set the following necessary environment variables in `train.sh`:  
```  
LOCAL_DIR=  
MODEL_PATH=  
```  
Set the required variables in `config/config.py`.  

## Running the Code  
Run the following command to start training:  
```  
bash train.sh  
```  
## Decode
```
bash infer.sh
```
# RoadMap  

## Notes  
- **Data Path**: Ensure that `train_data_path` and `eval_data_path` point to the correct data directories.  
- **Device Selection**: Based on your hardware environment, set `device_type` to `npu` or `cuda`. If not using NPU, comment out all `import torch_npu` statements. If you encounter any errors related to NPU, modify them to `cuda`.  
- **Dependency Installation**: Ensure all dependencies are correctly installed. If not installed, you can use the following command:  
```  
pip install numpy==1.26.0 torch==2.1.0 torch-npu==2.1.0.post10 torchaudio==2.1.0 torchvision==0.16.0  
```