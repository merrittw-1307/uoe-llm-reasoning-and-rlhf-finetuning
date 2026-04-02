import pandas as pd
import itertools
import numpy as np

steps = [1, 2, 3, 4]

def get_included_steps(row, steps):
    included_steps = []
    for step in steps:
        column_name = f'step{step}_present'
        if row[column_name] == 1:
            included_steps.append(step)
    return tuple(sorted(included_steps))

def generate_all_subsets(steps):
    all_subsets_included = []
    for r in range(len(steps) + 1):
        subsets_r = list(itertools.combinations(steps, r))
        all_subsets_included.extend(subsets_r)
    return all_subsets_included

def compute_v_S(df, all_subsets_included):
    v_S = {}
    for subset in all_subsets_included:
        subset_df = df[df['present_steps'] == subset]
        if not subset_df.empty:
            v_S[subset] = subset_df['is_correct'].mean()
        else:
            v_S[subset] = np.nan
    return v_S

def compute_marginal_contributions(steps, v_S):
    permutations = list(itertools.permutations(steps))
    Delta_sum = {i: 0.0 for i in steps}
    valid_permutations_count = 0
    total_steps_set = set(steps)
    for pi in permutations:
        valid_permutation = True
        for i in steps:
            idx_i = pi.index(i)
            S_i = pi[:idx_i]
            S_i_union_i = pi[:idx_i + 1]
            included_S_i_sorted = tuple(sorted(S_i))
            included_S_i_union_i_sorted = tuple(sorted(S_i_union_i))
            v_S_i = v_S.get(included_S_i_sorted, np.nan)
            v_S_i_union_i = v_S.get(included_S_i_union_i_sorted, np.nan)
            if np.isnan(v_S_i) or np.isnan(v_S_i_union_i):
                valid_permutation = False
                break
            else:
                Delta_sum[i] += (v_S_i_union_i - v_S_i)
        if valid_permutation:
            valid_permutations_count += 1
    return Delta_sum, valid_permutations_count

def compute_shapley_values(Delta_sum, valid_permutations_count, steps):
    if valid_permutations_count == 0:
        return {i: 0.0 for i in steps}
    shapley_values = {i: Delta_sum[i] / valid_permutations_count for i in steps}
    return shapley_values

def main():
    df = pd.read_csv('evaluation_with_steps.csv')
    df['present_steps'] = df.apply(get_included_steps, axis=1, args=(steps,))
    all_subsets_included = generate_all_subsets(steps)
    v_S = compute_v_S(df, all_subsets_included)
    Delta_sum, valid_count = compute_marginal_contributions(steps, v_S)
    shapley_values = compute_shapley_values(Delta_sum, valid_count, steps)
    print("-" * 30)
    print(f"Valid Count: {valid_count}")
    for step_num, val in shapley_values.items():
        print(f"Step {step_num} Shapley Value: {val:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()