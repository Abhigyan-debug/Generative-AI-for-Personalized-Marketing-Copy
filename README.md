# Generative AI for Personalized Marketing Copy

College assignment: build a pipeline that writes personalized marketing emails using an
open-source LLM, based on a customer's demographics and what they've bought before. Everything
runs locally - no OpenAI key, no paid API, just a Hugging Face model on your own machine.

Feed it a customer (age, city, favorite category, purchase history) and it writes a subject line,
greeting, body, product recommendation, promo offer, and CTA, in one of four styles - Professional,
Friendly, Luxury, Festive. It also scores each email on a few metrics (readability, sentiment,
personalization) so you're not just eyeballing the output and guessing if it's any good.

There are three ways to run it: a Jupyter notebook (`notebooks/development.ipynb`) that goes
through the pipeline step by step, a CLI (`main.py`) for batch-generating a whole campaign to CSV,
and a Streamlit app (`app.py`) if you want something click-through-able.

## What it does

It loads the customer and purchase history CSVs, checks they've got the right columns, and drops
the obviously bad rows (negative prices, dates it can't parse). The two get merged into one
profile per customer - age group, favorite category, total spend, loyalty tier, last thing they
bought - and a product gets recommended using TF-IDF + cosine similarity against that customer's
own purchase history, instead of just grabbing something random from their favorite category.

From there it builds a prompt from a template and a style guide, fills in the customer's details,
and sends it to a local LLM (default `microsoft/Phi-3-mini-4k-instruct`). The reply gets parsed
into six labeled parts, falling back to generic text for anything it can't find so one weird
generation doesn't take down the whole batch. Each email gets scored - word count, a hand-rolled
Flesch readability score, TextBlob sentiment, a personalization score - and everything gets
exported to CSV. The Streamlit app sits on top of all of that, with single-customer or batch mode,
CSV upload, and a small analytics tab for KMeans customer segments and a sentiment breakdown.

## Stack

Python, pandas, Hugging Face `transformers` + `torch` for the LLM, scikit-learn for the
recommender and the KMeans segmentation, TextBlob for sentiment, Streamlit + Altair for the app,
Jupyter for the notebook. Default model is Phi-3 Mini since it's small enough to actually run on a
laptop CPU, but Gemma / Mistral / Llama instruct variants work too - just pass a different
`--model`.

## Setup

```bash
git clone <this-repo-url>
cd generative_AI_email
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Sample data's already sitting in `data/`, nothing else to configure. First time you actually
generate something, `transformers` pulls the model from Hugging Face and caches it
(`~/.cache/huggingface`) - Phi-3 Mini is around 7-8 GB so that first download takes a bit. After
that it loads from disk. If you just want to test the pipeline quickly without waiting on a big
download, point `--model` at something small like `Qwen/Qwen2.5-0.5B-Instruct` (copy quality drops
a lot with tiny models, but it proves the plumbing works).

## Running it

Notebook, walks through everything step by step:

```bash
jupyter notebook notebooks/development.ipynb
```

CLI, batch-generates to CSV:

```bash
python main.py --style friendly --limit 5
```

`--limit` caps how many customers get processed - useful since this is CPU inference and slow.
Drop it to run the full list. You can also point it at your own model/data:

```bash
python main.py --style luxury --model microsoft/Phi-3-mini-4k-instruct --customers data/customers.csv --purchases data/purchase_history.csv
```

Output goes to `outputs/generated_emails_<timestamp>.csv`.

Streamlit app:

```bash
streamlit run app.py
```

## Folder structure

```
generative_AI_email/
├── app.py                        # Streamlit web app
├── main.py                       # CLI entry point
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── development.ipynb
├── src/
│   ├── config.py                 # shared paths + constants
│   ├── data_loader.py            # CSV loading + validation
│   ├── preprocessing.py          # merge, age groups, loyalty tiers, KMeans segments
│   ├── recommender.py            # TF-IDF content-based product recommender
│   ├── prompt_builder.py         # fills the prompt template per customer/style
│   ├── email_generator.py        # HF pipeline wrapper + section parsing
│   ├── evaluation.py             # word count, readability, sentiment, personalization
│   ├── export_utils.py           # CSV export
│   └── pipeline.py               # ties the above into one generate_campaign() call
├── prompts/
│   ├── base_prompt_template.txt
│   └── style_guides.json
├── data/
│   ├── customers.csv
│   ├── purchase_history.csv
│   └── product_catalog.csv
└── outputs/                      # generated CSVs land here (gitignored)
```

## What it actually outputs

This is a real run, `python main.py --limit 2 --style friendly`, default model, for a customer
whose last order was a Bluetooth speaker - not cherry-picked, just the first thing it generated:

> **Subject:** Boost Your Workout with Our Latest Fitness Gadget
>
> Hey Ishita,
>
> We noticed you're all about keeping up with tech trends, like that fantastic Portable Bluetooth
> Speaker you got. Speaking of which, staying active is just as trendy, and we've got something
> that'll sync perfectly with your tech lifestyle.
>
> **Recommendation:** Our Smart Fitness Band tracks your activity, connects to your devices, and
> it's as sleek as your current gadgets.
>
> **Offer:** Plus, enjoy free shipping on your next purchase - it's our way of saying thanks!
>
> **CTA:** Shop now and keep the tech momentum going!

Word count 89, readability 66.1 (fairly easy), sentiment positive, personalization score 25%.

That last number looks off for an email that clearly references a real past purchase - that's the
metric being dumb, not the copy. It's literal keyword matching (does the customer's name/city/
category/tier show up word-for-word), so it completely misses personalization that comes through
as natural language instead of restated fields. Didn't get around to fixing that, it's on the list
below.

Since generation is sampled and not greedy, running this yourself will give different wording.

## Things I'd still want to fix

Ran out of time before getting to these, but two things actually bug me. The personalization
score is pure keyword matching, so an email can reference the customer's exact last purchase in
a normal sentence and still score low just because it didn't repeat the field verbatim - that's
what happened in the example above, 25% for an email that's clearly personalized. And the regex
parser that splits the model's output into six sections just gives up and falls back to generic
text the moment a model doesn't follow the labeled format, instead of retrying or repairing what
it got.

## Troubleshooting

Hit `OSError: [WinError 1114] ... c10.dll` on Windows? That's a DLL load-order clash between torch
and scikit-learn/scipy that showed up during dev. It's worked around in `src/__init__.py` (torch
gets imported before anything else in the package), but if it comes back, importing `torch` before
`sklearn`/`scipy` anywhere in the process fixes it.

If generation is crawling and you see something about parameters being offloaded to disk, that's
`accelerate` deciding your machine doesn't have enough free RAM and swapping model layers to disk
instead - technically works, just really slow. `email_generator.py` avoids this on CPU-only setups
by skipping `device_map="auto"` unless a GPU is actually available. Still seeing it? Close a few
memory-hungry apps and try again.
