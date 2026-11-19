"""
Reflector core: prompt builders, label-only inference, and confidence regimes.

Copied verbatim from ``reflector/Reflection_Policy_Prompt_Grid.ipynb`` (cells 8 and 12) so that
the combined Router -> RAG -> Reflector pipeline uses exactly the same inference code as the
Reflector experiments. The only addition is ``build_messages_reflection_with_context``, which
puts retrieved reference text into a reflection prompt using the same wording as
``r2r_baseline_comparison.py``.
"""

from __future__ import annotations

import json
import re

import numpy as np

PROMPT_STYLE_ORDER = [
    "plain_retry",
    "error_scan",
    "option_elimination",
    "evidence_reconstruction",
]
CONTEXT_SOURCES = {"ReClor"}
RAG_INSTRUCTION = (
    "Use the provided reference information if relevant, "
    "otherwise rely on your own knowledge. "
)


def normalize_label(text: str) -> str:
    text = str(text).strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)
    if text and text[0].isalpha():
        return text[0]
    return text


def safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def get_passage_for_row(source: str, raw_context) -> str | None:
    if source not in CONTEXT_SOURCES:
        return None
    if raw_context is None:
        return None
    text = str(raw_context).strip()
    if text in {"", "nan", "None"}:
        return None
    return text


def build_qa_user_prompt(question: str, options: dict, passage: str | None = None,
                         rag_context: str | None = None) -> str:
    option_lines = [f"{label}. {text}" for label, text in options.items()]
    # rag_block matches r2r_baseline_comparison.build_user_prompt; empty when no context.
    rag_block = f"Reference Information:\n{rag_context}\n\n" if rag_context else ""
    passage_block = f"Passage:\n{passage}\n\n" if passage else ""
    return (
        f"{rag_block}{passage_block}"
        f"Question:\n{question}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nAnswer with only the label."
    )


def build_system_prompt_direct(valid_labels: list[str], with_rag: bool = False) -> str:
    label_list = ", ".join(valid_labels)
    rag_instr = RAG_INSTRUCTION if with_rag else ""
    return (
        "You are a multiple-choice question answering system. "
        f"{rag_instr}"
        "Read the question and options carefully. "
        f"Return ONLY the final answer label, one of {label_list}. "
        "Do not explain your reasoning. Do not write anything else."
    )


def build_system_prompt_plain_retry(valid_labels: list[str]) -> str:
    label_list = ", ".join(valid_labels)
    return (
        "You are a multiple-choice question answering system. "
        "Evaluate the problem from scratch with a fresh pass. "
        f"Return ONLY the final answer label, one of {label_list}. "
        "Do not explain your reasoning. Do not write anything else."
    )


def build_system_prompt_error_scan(valid_labels: list[str]) -> str:
    label_list = ", ".join(valid_labels)
    return (
        "You are a multiple-choice question answering system. "
        "Read the problem carefully and silently scan for likely mistakes such as negation, "
        "option mismatch, unsupported assumptions, or missed qualifiers. "
        f"Return ONLY the final answer label, one of {label_list}. "
        "Do not explain your reasoning. Do not write anything else."
    )


def build_system_prompt_option_elimination(valid_labels: list[str]) -> str:
    label_list = ", ".join(valid_labels)
    return (
        "You are a multiple-choice question answering system. "
        "Silently eliminate the weakest options first, then choose the strongest remaining answer. "
        f"Return ONLY the final answer label, one of {label_list}. "
        "Do not explain your reasoning. Do not write anything else."
    )


def build_system_prompt_evidence_reconstruction(valid_labels: list[str]) -> str:
    label_list = ", ".join(valid_labels)
    return (
        "You are a multiple-choice question answering system. "
        "Silently reconstruct the key evidence from the question and passage before deciding. "
        f"Return ONLY the final answer label, one of {label_list}. "
        "Do not explain your reasoning. Do not write anything else."
    )


REFLECTION_PROMPTS = {
    "plain_retry": build_system_prompt_plain_retry,
    "error_scan": build_system_prompt_error_scan,
    "option_elimination": build_system_prompt_option_elimination,
    "evidence_reconstruction": build_system_prompt_evidence_reconstruction,
}


def build_messages_direct(question, options, passage=None, rag_context=None) -> list[dict]:
    return [
        {"role": "system",
         "content": build_system_prompt_direct(list(options.keys()), with_rag=bool(rag_context))},
        {"role": "user", "content": build_qa_user_prompt(question, options, passage, rag_context)},
    ]


def build_messages_reflection(question, options, prompt_style, passage=None) -> list[dict]:
    return [
        {"role": "system", "content": REFLECTION_PROMPTS[prompt_style](list(options.keys()))},
        {"role": "user", "content": build_qa_user_prompt(question, options, passage)},
    ]


def build_messages_reflection_with_context(question, options, prompt_style, passage, rag_context):
    """NEW: reflection prompt + retrieved reference block (RAG instruction inserted after role line)."""
    system = REFLECTION_PROMPTS[prompt_style](list(options.keys()))
    head = "You are a multiple-choice question answering system. "
    system = head + RAG_INSTRUCTION + system[len(head):]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_qa_user_prompt(question, options, passage, rag_context)},
    ]


def build_prompt_text(messages: list[dict], tok) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_label_token_ids(label: str, tok) -> list[int]:
    token_ids = []
    for candidate in [label, f" {label}"]:
        ids = tok(candidate, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            token_ids.append(ids[0])
    token_ids = list(dict.fromkeys(token_ids))
    if not token_ids:
        raise ValueError(f"Label {label!r} has no single-token encoding.")
    return token_ids


def extract_first_token_signals(gen_out, options: dict, tok) -> dict:
    import torch

    option_labels = list(options.keys())
    step_scores = gen_out.scores[0][0]
    probs_vocab = torch.softmax(step_scores, dim=-1)
    log_probs_vocab = torch.log_softmax(step_scores, dim=-1)
    entropy_vocab = float(-(probs_vocab * torch.log(probs_vocab + 1e-12)).sum().item())

    label_scores = {}
    for label in option_labels:
        candidate_ids = get_label_token_ids(label, tok)
        candidate_lps = torch.tensor(
            [log_probs_vocab[token_id].item() for token_id in candidate_ids],
            dtype=torch.float32,
        )
        label_scores[label] = torch.logsumexp(candidate_lps, dim=0).item()

    score_tensor = torch.tensor([label_scores[label] for label in option_labels], dtype=torch.float32)
    label_probs_tensor = torch.softmax(score_tensor, dim=0)
    label_prob_dict = {label: float(label_probs_tensor[i].item()) for i, label in enumerate(option_labels)}

    sorted_labels = sorted(label_prob_dict.items(), key=lambda item: item[1], reverse=True)
    top1_label, top1_prob = sorted_labels[0]
    top2_label, top2_prob = sorted_labels[1] if len(sorted_labels) > 1 else (None, None)
    margin = float(top1_prob - top2_prob) if top2_prob is not None else np.nan
    entropy_labels = float(-(label_probs_tensor * torch.log(label_probs_tensor + 1e-12)).sum().item())

    return {
        "label_scores": label_scores,
        "label_probs": label_prob_dict,
        "top1_label": top1_label,
        "top1_prob": float(top1_prob),
        "top2_label": top2_label,
        "top2_prob": safe_float(top2_prob),
        "margin": safe_float(margin),
        "entropy_over_labels": safe_float(entropy_labels),
        "entropy_over_vocab": safe_float(entropy_vocab),
    }


def infer_label_only(messages: list[dict], options: dict, mdl, tok) -> dict:
    import torch

    prompt_text = build_prompt_text(messages, tok)
    inputs = tok(prompt_text, return_tensors="pt").to(mdl.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        gen_out = mdl.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.eos_token_id,
        )

    generated_text = tok.decode(gen_out.sequences[0][prompt_len:], skip_special_tokens=True).strip()
    generated_label = normalize_label(generated_text)
    signals = extract_first_token_signals(gen_out, options, tok)
    return {"generated_label": generated_label, "predicted_by_probs": signals["top1_label"], **signals}


def compute_blind_thresholds(direct_df) -> dict:
    valid = direct_df[direct_df["status"] == "ok"].copy() if "status" in direct_df else direct_df
    return {
        "entropy_hi": float(np.percentile(valid["entropy_over_labels"], 75)),
        "entropy_ok": float(np.percentile(valid["entropy_over_labels"], 50)),
        "margin_lo": float(np.percentile(valid["margin"], 25)),
        "margin_ok": float(np.percentile(valid["margin"], 50)),
    }


def assign_confidence_regime(entropy: float, margin: float, thresholds: dict) -> str:
    entropy = safe_float(entropy)
    margin = safe_float(margin)
    if np.isnan(entropy) or np.isnan(margin):
        return "unknown"
    if entropy <= thresholds["entropy_ok"] and margin >= thresholds["margin_ok"]:
        return "high"
    if entropy >= thresholds["entropy_hi"] or margin <= thresholds["margin_lo"]:
        return "low"
    return "medium"
