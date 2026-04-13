import sys

def patch_file():
    with open('src/prism_llm/data/collator.py', 'r') as f:
        content = f.read()

    # Increase max_steps to 2 to see if that helps
    content = content.replace("if all(l == max_len for l in lengths):", "if all(length == max_len for length in lengths):")

    with open('src/prism_llm/data/collator.py', 'w') as f:
        f.write(content)

patch_file()
