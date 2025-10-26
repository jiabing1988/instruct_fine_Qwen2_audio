import torch
import os
import json
import numpy as np
import kaldiio
import copy
import soundfile

class AudioDatset(torch.utils.data.Dataset):
    def __init__(self, data_path, prompt_path=None, wav_type="wav", inference_mode=False, max_duration=15.0):
        """
        data_path: 数据目录，需包含 daic.scp 和 daic_multitask.jsonl
        prompt_path: 包含任务 prompt 的 jsonl 文件路径
        wav_type: 'wav' 或 'ark'
        inference_mode: True 时不返回 target，用于推理
        max_duration: 最大音频时长（秒），超过将截断
        """
        self.wav_scp = {}
        self.tasks = []
        self.prompt = {}
        self.wav_type = wav_type
        self.inference_mode = inference_mode
        self.max_duration = max_duration  # 最大允许时长（秒）

        # 读取 wav.scp
        with open(os.path.join(data_path, "daic.scp")) as f:
            for line in f:
                utt_id, wav_path = line.strip().split(" ", 1)
                self.wav_scp[utt_id] = wav_path

        # 读取任务文件
        with open(os.path.join(data_path, "daic_multitask.jsonl")) as f:
            for line in f:
                self.tasks.append(json.loads(line))

        # 读取 prompt 文件
        with open(os.path.join(prompt_path)) as f:
            for line in f:
                item = json.loads(line)
                self.prompt[item["task"]] = item["prompt"]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # 获取任务信息
        key = self.tasks[idx]["key"]
        target = self.tasks[idx]["target"]
        prompt = self.prompt[self.tasks[idx]["task"]]

        # 读取音频
        if self.wav_type == "ark":
            audio = kaldiio.load_mat(self.wav_scp[key])[1].astype(np.float32) / 32768
            sr = 16000  #数据采样率为 16kHz
        elif self.wav_type == "wav":
            audio, sr = soundfile.read(self.wav_scp[key])
            if len(audio.shape) > 1:
                # 如果是多声道，取第一个声道
                audio = audio[:, 0]
        else:
            raise ValueError(f"Unsupported wav_type: {self.wav_type}")

        # # ===== 截断逻辑 =====
        # max_samples = int(self.max_duration * sr)
        # if len(audio) > max_samples:
        #     if not self.inference_mode:
        #         # 训练模式：随机截取一段音频
        #         start = np.random.randint(0, len(audio) - max_samples)
        #         audio = audio[start:start + max_samples]
        #     else:
        #         # 推理模式：取开头 15 秒
        #         audio = audio[:max_samples]

        # ===== 返回数据 =====
        if not self.inference_mode:
            return {
                "prompt": prompt,
                "audio": audio,
                "target": target
            }
        else:
            return {
                "prompt": prompt,
                "audio": audio,
                "target": "",
                "key": key
            }


def collate_fn_qwen2audio(samples, processor):
    prompt = [_["prompt"] for _ in samples]
    audio = [_["audio"] for _ in samples]
    target = [_["target"] for _ in samples]

    # 拼接 prompt + target 形成完整输入文本
    processed_data = processor(
        text=[i + j for i, j in zip(prompt, target)],
        audios=audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True
    )

    # ===== 处理 labels（mask 掉 prompt 部分）=====
    labels = copy.deepcopy(processed_data["input_ids"])
    text_ids = processor(prompt, return_tensors="pt", padding=True)

    for i, attention_mask in enumerate(text_ids["attention_mask"]):
        labels[i, :sum(attention_mask) +
               (processed_data["input_ids"][i] == processor.tokenizer.pad_token_id).sum().item()] = -100

    processed_data["labels"] = labels

    if "key" in samples[0]:
        keys = [_["key"] for _ in samples]
        processed_data["keys"] = keys

    return processed_data
