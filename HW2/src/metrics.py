def MAP(result, answer, k=100):
    result = result[:100]
    answer = answer[:100]
    correct_cnt = 0
    score = 0
    for i, r in enumerate(result, 1):
        if r in answer:
            correct_cnt += 1
            score += correct_cnt / i

    return score / max(min(len(result), len(answer)), 1)
