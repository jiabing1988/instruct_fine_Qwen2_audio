import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import get_peft_model, LoraConfig
from utils.set_seed import set_seed
from utils.set_logger import set_logger
from utils.functions import compute_acc, compute_metrics
from src.dataset import AudioDatset, collate_fn_qwen2audio
from transformers.modeling_outputs import BaseModelOutput
import gc


# ==========================
# GPU 内存监控
# ==========================
def print_gpu_memory(desc=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{desc} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


# ==========================
# Adapter 定义
# ==========================
class DepAdapter(nn.Module):
    def __init__(self, audio_dim, adapter_dim=512, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(audio_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, audio_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(audio_dim)
        
    def forward(self, audio_features):
        residual = audio_features
        x = self.down_proj(audio_features)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


def create_modified_qwen2audio_encoder(original_encoder, adapter_config):
    """为 Qwen2Audio Encoder 添加 Adapter"""
    audio_dim = original_encoder.config.d_model
    adapter = DepAdapter(
        audio_dim=audio_dim,
        adapter_dim=adapter_config.get("adapter_dim", 512),
        dropout=adapter_config.get("dropout", 0.1)
    )
    original_forward = original_encoder.forward

    def new_forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 确保使用 return_dict=True
        if return_dict is None:
            return_dict = self.config.use_return_dict
            
        outputs = original_forward(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True  # 强制使用 return_dict=True
        )
        
        # 现在 outputs 一定是 BaseModelOutput
        audio_features = outputs.last_hidden_state
        adapted_audio_features = adapter(audio_features)
        
        # 返回 BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=adapted_audio_features,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    # 绑定新的 forward 方法
    original_encoder.forward = new_forward.__get__(original_encoder, type(original_encoder))
    original_encoder.audio_adapter = adapter
    return original_encoder


# ==========================
# 优化的单卡训练函数
# ==========================
def train_single():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置内存优化
    torch.cuda.empty_cache()
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # 启用内存高效注意力
    
    model_path = "/root/autodl-tmp/Qwen/Qwen2-Audio-7B-Instruct"
    save_path = f"/root/autodl-tmp/Qwen2-Audio-finetune/output_model/single-{time.strftime('%H-%M-%S')}"
    train_data_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/train"
    eval_data_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/val"
    train_prompt_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/train/daic_multiprompt.jsonl"
    eval_prompt_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/val/daic_multiprompt.jsonl"
    wav_type = "wav"

    os.makedirs(save_path, exist_ok=True)

    logger = set_logger(save_path)
    set_seed(1234)

    print_gpu_memory("初始状态")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 减少内存占用
        low_cpu_mem_usage=True,      # 减少 CPU 内存使用
    )
    
    print_gpu_memory("模型加载后")

    # ========== 添加 Adapter 到 audio_tower ==========
    adapter_config = {"adapter_dim": 256, "dropout": 0.1}
    model.audio_tower = create_modified_qwen2audio_encoder(model.audio_tower, adapter_config)

    # ========== 仅给 language_model 添加 LoRA ==========
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 只在 language_model 上注入 LoRA，而不是整个模型
    if hasattr(model, "language_model"):
        model.language_model = get_peft_model(model.language_model, lora_config)
    else:
        raise RuntimeError("模型中未找到 language_model 属性，请检查结构。")

    # ========== 冻结所有参数，只解冻 audio_adapter、multi_modal_projector 和 LoRA ==========
    for name, p in model.named_parameters():
        p.requires_grad = False

    # 解冻 adapter 参数
    if hasattr(model.audio_tower, "audio_adapter"):
        for n, p in model.audio_tower.audio_adapter.named_parameters():
            p.requires_grad = True
            print(f"解冻 audio_adapter 参数: {n}")

    # 解冻 multi_modal_projector 参数
    # if hasattr(model, "multi_modal_projector"):
    #     for n, p in model.multi_modal_projector.named_parameters():
    #         p.requires_grad = True
    #         print(f"解冻 multi_modal_projector 参数: {n}")
    # else:
    #     print("未找到 multi_modal_projector 模块")

    # 解冻 LoRA 参数
    for name, p in model.named_parameters():
        if "lora" in name.lower() or "peft" in name.lower():
            p.requires_grad = True
            print(f"解冻 LoRA 参数: {name}")

    # 检查是否仍有 audio_tower 的 lora 参数（应当没有）
    for name, p in model.named_parameters():
        if "audio_tower" in name and "audio_adapter" not in name and p.requires_grad:
            print(f"Unexpected trainable param in audio_tower: {name}")

    # 打印可训练参数统计
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params_count:,}")
    print(f"可训练参数比例: {trainable_params_count/total_params*100:.4f}%")

    # 启用梯度检查点（显著减少内存占用）
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("已启用梯度检查点")

    model.to(device)
    print_gpu_memory("模型移动到设备后")

    # 优化器与调度器
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"可训练参数数量: {len(trainable_params)}")
    
    optim = torch.optim.AdamW(trainable_params, lr=1e-4)
    total_train_steps = 1000
    warmup_steps = 100
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(step / warmup_steps, 1.0)
        if step < warmup_steps
        else max(0.0, 1 - (step - warmup_steps) / (total_train_steps - warmup_steps))
    )

    # 数据加载
    train_dataset = AudioDatset(train_data_path, train_prompt_path, wav_type)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor)
    )
    eval_dataset = AudioDatset(eval_data_path, eval_prompt_path, wav_type)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor)
    )

    # ==========================
    # 优化的训练循环
    # ==========================
    best_f1 = -float("inf")
    
    for epoch in range(10):
        model.train()
        train_bar = tqdm(train_loader, desc=f"[Train] epoch {epoch}")
        
        for step, batch in enumerate(train_bar):
            # 清理前一步的缓存
            if step > 0 and step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 前向传播（使用混合精度，但不使用梯度缩放）
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                acc = compute_acc(outputs.logits, batch["labels"])

            # 反向传播（不使用梯度缩放）
            optim.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optim.step()
            scheduler.step()

            train_bar.set_description(f"[Train] epoch {epoch} | loss {loss:.3f} | acc {acc:.3f}")
            
            # 每50步打印一次内存使用情况
            if step % 50 == 0:
                print_gpu_memory(f"训练步骤 {step}")

        # 验证阶段 - 修正指标计算
        model.eval()
        eval_loss, eval_acc, eval_f1, eval_precision, eval_recall = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            eval_bar = tqdm(eval_loader, desc="[Eval]")
            for step, batch in enumerate(eval_bar):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 验证阶段也使用混合精度
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accuracy, precision, recall, f1 = compute_metrics(outputs.logits, batch["labels"])
                
                eval_loss += loss.item()
                eval_acc += accuracy.item()
                eval_f1 += f1.item()
                eval_precision += precision.item()
                eval_recall += recall.item()
                
                # 显示当前 batch 的指标
                eval_bar.set_description(f"[Eval] loss {loss:.3f} | acc {accuracy:.4f} | f1 {f1:.4f} | pre {precision:.4f} | rec {recall:.4f}")

        # 计算平均指标
        num_eval_batches = len(eval_loader)
        eval_loss /= num_eval_batches
        eval_acc /= num_eval_batches
        eval_f1 /= num_eval_batches
        eval_precision /= num_eval_batches
        eval_recall /= num_eval_batches

        # 记录平均指标到日志
        logger.info(f"[Eval] epoch {epoch} | loss={eval_loss:.3f}, acc={eval_acc:.4f}, f1={eval_f1:.4f}, pre={eval_precision:.4f}, rec={eval_recall:.4f}")

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            # 只保存可训练的参数
            model.save_pretrained(save_path, safe_serialization=True)
            processor.save_pretrained(save_path)
            logger.info(f"[Saving Best Model] F1 improved to {best_f1:.3f} → {save_path}")
        
        # 清理验证阶段的内存
        torch.cuda.empty_cache()
        gc.collect()

    print(f"训练完成，最佳 F1 = {best_f1:.3f}")
    print_gpu_memory("训练完成后")


if __name__ == "__main__":
    train_single()