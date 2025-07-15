import ast

input_path = "acc.txt"
output_path = "true_aug_results.txt"

with open(input_path, "r") as f, open(output_path, "w") as out_f:
    for line in f:
        if not line.strip():
            continue
        config_str, acc_str = line.strip().split("},")
        config = ast.literal_eval(config_str + "}")
        acc = acc_str.split(":")[1].strip()
        true_augs = [k for k, v in config.items() if v]
        out_f.write(f"True augmentations: {true_augs}, Accuracy: {acc}\n")
