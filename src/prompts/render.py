from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('src/prompts'))
template = env.get_template('classifier.jinja2')

def prepare_prompt(categories, examples, input):
    return template.render(
        categories=categories,
        examples=examples,
        input=input
    )


if __name__ == "__main__":
    categories = ["credit_reporting", "mortgages_and_loans", "credit_card", "retail_banking"]

    examples = [
        {
            "text": "I noticed an error on my credit report regarding a debt I already paid off.", 
            "category": "credit_reporting"
        },
        {
            "text": "What are the current interest rates for a fixed-rate 30-year home loan?", 
            "category": "mortgages_and_loans"
        },
        {
            "text": "I lost my physical card and need to block it and request a replacement immediately.", 
            "category": "credit_card"
        },
        {
            "text": "I need to check my checking account balance and transfer money to another local bank.", 
            "category": "retail_banking"
        }
    ]

    input = "I want to apply for a line of credit to buy my first apartment next month."
    prompt = prepare_prompt(categories, examples, input)

    print(prompt)