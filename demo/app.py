"""Gradio web demo:  python demo/app.py  ->  http://localhost:7860

Type a video caption, get hashtags from RoBERTa + RL (or the 14 MB Keras Transformer
if RoBERTa has not been trained locally); tick
"LLM polish" to send the candidates to an LLM (needs ANTHROPIC_API_KEY,
OPENAI_API_KEY, or a local Ollama server).
"""
import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger.llm import prompts  # noqa: E402
from hashtagger.llm.client import LLMClient, parse_json  # noqa: E402
from hashtagger.predict import Hashtagger  # noqa: E402
from hashtagger.utils import ROOT  # noqa: E402

H = Hashtagger.load()
PROVIDER = ("anthropic" if os.environ.get("ANTHROPIC_API_KEY")
            else "openai" if os.environ.get("OPENAI_API_KEY") else "ollama")
MODEL = os.environ.get("HASHTAG_LLM_MODEL", "claude-opus-5-5" if PROVIDER == "anthropic" else "llama3.1")
EXAMPLES = [
    "a chef slices onions and fries them in a pan",
    "two teams compete in a basketball game in front of a crowd",
    "a woman applies eye makeup in front of a mirror",
    "kids playing minecraft and building a house",
    "a red sports car drives fast on a mountain road",
    "a news anchor reports on the presidential election",
]


def run(caption: str, use_llm: bool):
    if not caption.strip():
        return "", {}, ""
    tags = H.predict(caption)
    cat, _ = H.category(caption)
    model_line = " ".join(f"#{t}" for t, _ in tags)
    conf = {f"#{t}": p for t, p in H.candidates(caption, 10)}
    llm_line = ""
    if use_llm:
        try:
            client = LLMClient(PROVIDER, MODEL, ROOT / "data" / "llm_cache" / "demo.jsonl")
            out = parse_json(client.complete(prompts.SYSTEM, prompts.hybrid(caption, H.candidates(caption),
                                                                            H.category_tags)))
            llm_line = " ".join("#" + t for t in out.get("hashtags", []))
            if out.get("creative"):
                llm_line += "   ✨ " + " ".join("#" + str(t).lstrip("#") for t in out["creative"])
        except Exception as e:  # noqa: BLE001
            llm_line = f"(LLM unavailable: {e})"
    return f"**Category:** #{cat}\n\n**Hashtags:** {model_line}", conf, llm_line


with gr.Blocks(title="Video Hashtag Generator") as demo:
    gr.Markdown(f"# 🎬 Video Hashtag Generator\nTagger: **{H.kind}**, optional LLM re-ranking. "
                "Trained on MSR-VTT (10k videos, 20 categories).")
    with gr.Row():
        with gr.Column():
            cap = gr.Textbox(label="Video caption / description", lines=2)
            llm = gr.Checkbox(label=f"LLM polish ({PROVIDER}: {MODEL})", value=False)
            btn = gr.Button("Generate hashtags", variant="primary")
            gr.Examples(EXAMPLES, inputs=cap)
        with gr.Column():
            out_md = gr.Markdown()
            out_conf = gr.Label(label="Model confidence (top 10)", num_top_classes=10)
            out_llm = gr.Textbox(label="LLM re-ranked + creative tags")
    btn.click(run, [cap, llm], [out_md, out_conf, out_llm])
    cap.submit(run, [cap, llm], [out_md, out_conf, out_llm])

if __name__ == "__main__":
    demo.launch()
