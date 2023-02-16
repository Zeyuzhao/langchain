# Different ways to summarize
# Map & reduce into -> contiguous summary
# OR, can mapreduce -> structured knowledge. Need the creation of prompt, and then the structured extraction
# of information

# Example:

import openai


# summerize(): takes longer sentence -> into shorter sentence.

# Does map & reduce!

import os
import openai

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize(text: str):
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = prompt_template.format(text=text)
    # TODO: make call to openai, and parse the result


    res = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0
    )

    return res['choices'][0]['text']


def reduce_summarize(existing_summary: str, text: str):
    prompt_template = \
    """Your job is to produce a final summary. 
    EXISTING_SUMMARY:
    ---
    {existing_summary}
    ---
    
    NEW_CONTEXT:
    ---
    {text}
    ---
    
    Given the new context, refine the existing summary.
    NEW_SUMMARY:"""

    prompt = prompt_template.format(existing_summary=existing_summary, text=text)
    # TODO: make call to openai, and parse the result

    res = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0
    )

    return res['choices'][0]['text']


from pathlib import Path
filename = Path('docs/modules/state_of_the_union.txt')

with open(filename) as file:
    text = file.read()

# Do chunking for summerization.
chunks = text.split('\n\n')[:5]

# This part we can parallelize
summaries = [
    summarize(c) for c in chunks
]
print(summaries)

summary_acc = ""
for s in summaries:
    summary_acc = reduce_summarize(summary_acc, s)
    print(f'Intermediate state: ', summary_acc)
# Can call map_reduce on the result!

print(summary_acc)

