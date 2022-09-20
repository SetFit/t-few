from src.utils.Config import Config
from datasets import load_dataset
import numpy as np
from promptsource.templates import DatasetTemplates
from transformers import AutoTokenizer

def main(datasets):
    overall_seq_lens, overall_seq_lens_wo_prompts = [], []
    for dataset in datasets:
        config = Config(f"{dataset}.json")
        split = load_dataset(f"SetFit/{config.dataset}", split="train")
        prompts_dataset = config.dataset if config.prompts_dataset is None else config.prompts_dataset
        prompts = DatasetTemplates(prompts_dataset)

        seq_lens, seq_lens_wo_prompts = [], []
        # Select a prompt by its name
        for prompt in prompts.templates.values():
            # Tokenize
            tok = AutoTokenizer.from_pretrained("bigscience/T0")

            def compute_seq_len_wo_promtps(row):
                return {"seq_len_wo_prompts": len(tok(row["text"], max_length=256, truncation=True)["input_ids"])}

            # Compute without prompts
            split_wo_prompts = split.map(compute_seq_len_wo_promtps, num_proc=6)
            seq_lens_wo_prompts.extend(split_wo_prompts["seq_len_wo_prompts"])
            overall_seq_lens_wo_prompts.extend(split_wo_prompts["seq_len_wo_prompts"])

            ###############################################################################
            # Apply the prompt to the dataset
            def add_prompt(row):
                result = prompt.apply(row)
                return {"text": result[0] + "" + result[1]}

            # Compute with prompts
            split_w_prompts = split.map(add_prompt, num_proc=6)

            def compute_seq_len(row):
                return {"seq_len": len(tok(row["text"], max_length=256, truncation=True)["input_ids"])}

            # Compute
            split_w_prompts = split_w_prompts.map(compute_seq_len, num_proc=6)
            seq_lens.extend(split_w_prompts["seq_len"])
            overall_seq_lens.extend(split_w_prompts["seq_len"])

        with_prompts_str = f"{config.dataset}:\n median: {np.median(seq_lens)}, mean: {np.mean(seq_lens)})\n"
        print(with_prompts_str)

        wo_prompts_str = f"{config.dataset} - without prompts:\n median: {np.median(seq_lens_wo_prompts)}, mean: {np.mean(seq_lens_wo_prompts)})\n\n"
        print(wo_prompts_str)

        with open("seq_lens.txt", "a") as f:
            f.write(with_prompts_str)
            f.write(wo_prompts_str)

    with_prompts_str = f"\n\nAll datasets: \n median: {np.median(overall_seq_lens)}, mean: {np.mean(overall_seq_lens)})\n"
    print(with_prompts_str)

    wo_prompts_str = f"All datasets - without prompts:\n median: {np.median(overall_seq_lens_wo_prompts)}, mean: {np.mean(overall_seq_lens_wo_prompts)})\n"
    print(wo_prompts_str)

    with open("seq_lens.txt", "a") as f:
        f.write(with_prompts_str)
        f.write(wo_prompts_str)


if __name__ == "__main__":
    test_datasets = ["amazon_counterfactual_en", "emotion", "enron_spam", "SentEval-CR", "sst5"]
    main(test_datasets)