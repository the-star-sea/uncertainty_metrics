# lm_eval/metrics/singlepass_uncertainty.py
import numpy as np

def summarize_generation_scores(generated_token_ids, step_logits_list):
    """
    Summarize uncertainty from a single greedy decode.

    Args:
      generated_token_ids: 1D np.ndarray[int] of length T_gen (ids of generated tokens)
      step_logits_list: list length T_gen; each item is 1D np.ndarray[float] of length |V| (logits)

    Returns (all floats):
      mean_logprob, geo_prob (exp of mean_logprob in [0,1]),
      min_logprob, last_token_prob, mean_entropy, last_entropy
    """
    if not step_logits_list:
        return {
            "mean_logprob": float("nan"),
            "geo_prob": float("nan"),
            "min_logprob": float("nan"),
            "last_token_prob": float("nan"),
            "mean_entropy": float("nan"),
            "last_entropy": float("nan"),
        }

    logprobs = []
    entropies = []

    for t, logits in enumerate(step_logits_list):
        m = logits.max()
        exps = np.exp(logits - m)
        probs = exps / exps.sum()
        tok_id = int(generated_token_ids[t])
        p_tok = float(probs[tok_id]) if 0 <= tok_id < probs.shape[0] else 1e-12
        logprobs.append(np.log(p_tok + 1e-12))
        entropies.append(float(-(probs * np.log(probs + 1e-12)).sum()))

    logprobs = np.array(logprobs, dtype=np.float64)
    entropies = np.array(entropies, dtype=np.float64)

    mean_logprob = float(np.mean(logprobs))
    return {
        "mean_logprob": mean_logprob,
        "geo_prob": float(np.exp(mean_logprob)),
        "min_logprob": float(np.min(logprobs)),
        "last_token_prob": float(np.exp(logprobs[-1])),
        "mean_entropy": float(np.mean(entropies)),
        "last_entropy": float(entropies[-1]),
    }
