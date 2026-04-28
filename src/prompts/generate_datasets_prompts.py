import json
import pandas as pd
from src.prompts.render import prepare_prompt

def get_examples(df, categories):
    examples = []

    for cat in categories:
        sample = df[df["product"] == cat].sample(n=1)
        
        if not sample.empty:
            examples.append({
                "text": sample["narrative"].iloc[0],
                "category": sample["product"].iloc[0]
            })
            
            df = df.drop(sample.index)

    return examples


def render_prompts(df, categories, examples):
    prompts = []
    for row in df.itertuples():
        text = row.narrative
        prompt = prepare_prompt(categories, examples, text)
        prompts.append(prompt)
    
    return prompts


def save_as_jsonl(prompts, df_original, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts):
            data = {
                "custom_id": f"request_{i}",
                "prompt": prompt,
                "target_category": df_original.iloc[i]["product"]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    categories = ['credit_reporting', 'mortgages_and_loans', 'credit_card', 'retail_banking']
    df = pd.read_csv("data/long-500.csv")
    examples = get_examples(df, categories)
    prompts = render_prompts(df, categories, examples)
    save_as_jsonl(prompts, df, "data/long_500_prompts.jsonl")

    df = pd.read_csv("data/short-500.csv")
    examples = get_examples(df, categories)
    prompts = render_prompts(df, categories, examples)
    save_as_jsonl(prompts, df, "data/short_500_prompts.jsonl")
