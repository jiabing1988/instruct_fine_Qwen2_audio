import os

dataset_path = "/root/autodl-tmp/DAIC_Qwen2Audio/DAIC/data/daic_xcy/val"  
output_file = "/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/daic_val.scp"

with open(output_file, 'w', encoding='utf-8') as f:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                abs_path = os.path.abspath(os.path.join(root, file))
                file_id = os.path.splitext(file)[0]
                f.write(f"{file_id} {abs_path}\n")

print(f"SCP文件已生成: {output_file}")