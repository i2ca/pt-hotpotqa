from datasets import load_dataset, concatenate_datasets
import json

seed = 429029
n_sample = 400
outfile = "datasets/original_sample_set.json"

hotpot_qa = load_dataset("hotpot_qa", "distractor")

full_set = hotpot_qa['validation']
bridge_set = full_set.filter(lambda x: x["type"] == "bridge")
comparison_set = full_set.filter(lambda x: x["type"] == "comparison")

n_bridge = round(n_sample * len(bridge_set) / len(full_set))
n_comparison = n_sample - n_bridge

bridge_subset = bridge_set.shuffle(seed).select(range(n_bridge))
comparison_subset = comparison_set.shuffle(seed+1).select(range(n_comparison))

sample_set = concatenate_datasets([bridge_subset, comparison_subset]).shuffle(seed=seed+2)

with open(outfile, "w", encoding="utf-8") as f:
    json.dump(sample_set.to_list(), f, ensure_ascii=False, indent=4)

