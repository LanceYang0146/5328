import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def estimate_T_confident(model, loader, device, topk=100, num_classes=3):
    """Estimate transition matrix T via confident anchors.
    Steps:
      1) Get predicted probabilities p(y|x) from a model trained on noisy labels.
      2) For each class i (assumed to be *noisy* label), select 'topk' samples whose predicted class is i
         with highest confidence.
      3) Average the full probability vectors over those samples -> row i of T_hat.
    This approximates P(noisy=j | true=i) under anchor assumption.
    """
    model.eval()
    all_probs = []
    all_preds = []
    for x, _ in loader:  # we don't need labels here
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
    P = np.concatenate(all_probs, axis=0)
    yhat = np.concatenate(all_preds, axis=0)

    C = num_classes
    T_hat = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        idx = np.where(yhat == i)[0]
        if len(idx) == 0:
            # fallback: uniform row
            T_hat[i] = np.ones(C) / C
            continue
        conf = P[idx, i]
        top_idx = idx[np.argsort(-conf)[:min(topk, len(idx))]]
        T_hat[i] = P[top_idx].mean(axis=0)
        # normalize as probability
        s = T_hat[i].sum()
        if s <= 0:
            T_hat[i] = np.ones(C) / C
        else:
            T_hat[i] /= s
    return torch.tensor(T_hat, dtype=torch.float32)
