def evaluate(preds: list, gold_preds: list, mn_labels: list, verbose=False):
    """
    Returns prediction statistics: 
        1. % exact-match 
        2. % correct tokens
        3. % max-len correct span
    
    Each stat is broken down into:
        a. ACC: score for all sequences
        b. MN: score for monotonic sequences
        c. NMN: score for nonmonotonic sequences

    :param preds: list of model predictions
    :param gold_preds: list of ground truth meaning representations
    :param mn_labels: list of monotonic alignment labels
    :param verbose: bool if True the function prints the stats to std out

    Usage:
    >>> preds = ["answer len riverid colorado", "answer population cityid ...]
    >>> gold_preds = ["answer len riverid colorado", "answer density_1 ...]
    >>> mn_labels = [1, 0, 1, 1, 0, ...]
    >>> evaluate(preds, gold_preds, mn_labels)
    """
    stats = {}

    acc, mn_acc, nmn_acc = exact_match_acc(preds, gold_preds, mn_labels)
    stats["exact_match"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"EXACT-MATCH ACCURACY")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%\n")

    acc, mn_acc, nmn_acc = no_correct_tokens(preds, gold_preds, mn_labels)
    stats["no_correct_tokens"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"NO. CORRECT TOKENS")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%\n")

    acc, mn_acc, nmn_acc = max_correct_span(preds, gold_preds, mn_labels)
    stats["max_correct_span"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"MAX CORRECT SPAN")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%")
    
    return stats

def exact_match_acc(preds: list, gold_preds: list, mn_labels: list):
    
    mn_correct = nmn_correct = 0
    for pred, gold, mn in zip(preds, gold_preds, mn_labels):
        if pred == gold:
            if mn: mn_correct += 1
            else: nmn_correct += 1
        
    acc = mn_correct + nmn_correct
    acc *= 100
    acc /= len(preds)

    mn_acc = mn_correct * 100
    mn_acc /= sum(mn_labels)

    nmn_acc = nmn_correct * 100
    nmn_acc /= len(mn_labels) - sum(mn_labels)

    return acc, mn_acc, nmn_acc

def no_correct_tokens(preds: list, gold_preds: list, mn_labels: list):

    acc = mn_acc = nmn_acc = 0
    lens = mn_lens = nmn_lens = 0
    for pred, gold, mn in zip(preds, gold_preds, mn_labels):
        pred = pred.split()
        gold = gold.split()

        correct_tokens = 0
        for i in range(min(len(pred), len(gold))):
            if pred[i] == gold[i]: correct_tokens += 1
        
        acc += (correct_tokens * 100 / len(gold)) * len(gold)
        lens += len(gold)
        if mn: 
            mn_acc += (correct_tokens * 100 / len(gold)) * len(gold)
            mn_lens += len(gold)
        else: 
            nmn_acc += (correct_tokens * 100 / len(gold)) * len(gold)
            nmn_lens += len(gold)

    acc /= lens
    mn_acc /= mn_lens
    nmn_acc /= nmn_lens

    return acc, mn_acc, nmn_acc

def max_correct_span(preds: list, gold_preds: list, mn_labels: list):

    acc = mn_acc = nmn_acc = 0
    lens = mn_lens = nmn_lens = 0
    for pred, gold, mn in zip(preds, gold_preds, mn_labels):
        pred = pred.split()
        gold = gold.split()

        max_correct = 0
        for i in range(min(len(pred), len(gold))):
            if pred[i] != gold[i]: continue
            correct = 1
            rhs_len = min(len(pred)-i-1, len(gold)-i-1)
            for j in range(rhs_len):
                if pred[j] == gold[j]:
                    correct += 1
                else: break
            if correct > max_correct: max_correct = correct
        
        acc += (max_correct * 100 / len(gold)) * len(gold)
        lens += len(gold)
        if mn: 
            mn_acc += (max_correct * 100 / len(gold)) * len(gold)
            mn_lens += len(gold)
        else: 
            nmn_acc += (max_correct * 100 / len(gold)) * len(gold)
            nmn_lens += len(gold)
        
    acc /= lens
    mn_acc /= mn_lens
    nmn_acc /= nmn_lens

    return acc, mn_acc, nmn_acc