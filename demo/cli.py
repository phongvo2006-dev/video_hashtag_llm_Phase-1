"""Command-line demo.

    python demo/cli.py "a chef slices onions and fries them in a pan"
    python demo/cli.py --llm anthropic "kids playing minecraft and laughing"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger.predict import Hashtagger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("caption", nargs="+")
ap.add_argument("--llm", choices=["anthropic", "openai", "ollama"], help="add LLM re-ranking (needs key / server)")
ap.add_argument("--model", default="claude-opus-5-5")
args = ap.parse_args()
caption = " ".join(args.caption)

h = Hashtagger.load()
cat, pc = h.category(caption)
print(f'\nCaption : "{caption}"')
print(f"Category: #{cat}  ({pc:.2f})")
print("Tags    : " + "  ".join(f"#{t} ({p:.2f})" for t, p in h.predict(caption)))

if args.llm:
    from hashtagger.llm import prompts
    from hashtagger.llm.client import LLMClient, parse_json
    from hashtagger.utils import ROOT
    client = LLMClient(args.llm, args.model, ROOT / "data" / "llm_cache" / "demo.jsonl")
    out = parse_json(client.complete(prompts.SYSTEM, prompts.hybrid(caption, h.candidates(caption),
                                                                    h.category_tags)))
    print("LLM     : " + "  ".join("#" + t for t in out.get("hashtags", [])))
    print("Creative: " + "  ".join("#" + str(t).lstrip("#") for t in out.get("creative", [])))
