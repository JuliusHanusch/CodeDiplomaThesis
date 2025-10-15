from constants import *

def linear_regression(xs: pd.Series, ys: pd.Series) -> tuple[float, float]:
    model = LinearRegression()
    model.fit(xs.values.reshape(-1, 1), ys.values)
    return float(model.coef_[0]), float(model.intercept_)

def load_data(**kwargs):
    conn = sqlite3.Connection(db_path)
    results = pd.read_sql_query("SELECT * FROM Results", conn)
    conn.close()
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna()
    return results

def load_data_adjusted_for_poor_hp_choices(metric : str, **kwargs):
    conn = sqlite3.Connection(db_path)
    results = pd.read_sql_query("SELECT * FROM Results", conn)
    conn.close()
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna()
    results["parameter_bins"] = pd.qcut(results['parameters'], q=10)
    results = results.loc[results.groupby(["budget", "parameter_bins"], observed=True)[metric].idxmin()]
    return results

def calc_scaling_law(scaling_factor="parameters", metric="MASE", color=alt.Undefined, hp_adjusted=False, file_name="scale_law_size", title="Scaling Law with Parameter Count N", title_x="Parameters (non-embedding)"):
    if hp_adjusted:
        results = load_data_adjusted_for_poor_hp_choices(metric=metric)
    else:
        results = load_data()
    print(results)
    results["context_length"] = results["config"].apply(lambda x: 2**json.loads(x)["context_length_expo"])
    results["training_tokens"] = results["budget"] * results["context_length"]
    results["compute"] = 6 * results["parameters"] * results["training_tokens"]
    if color != alt.Undefined:
        results = results[[scaling_factor, metric, color.shorthand]]
    else:
        results = results[[scaling_factor, metric]]
    results[[f"{scaling_factor}_log"]] = results[[scaling_factor]].applymap(lambda x: np.log(x) if np.isscalar(x) and x > 0 else np.nan)
    results[["loss"]] = results[[metric]].applymap(lambda x: np.exp(x) if np.isscalar(x) and x > 0 else np.nan)

    slope, intercept = linear_regression(results[f"{scaling_factor}_log"], results[metric])

    results[f"E({metric})_pred"] = slope * results[f"{scaling_factor}_log"] + intercept
    results[f"loss_pred"] = np.exp(slope * results[f"{scaling_factor}_log"] + intercept)

    X = alt.X(
            scaling_factor, 
            scale=alt.Scale(
                type="log", 
                domain=(min(results[scaling_factor]), max(results[scaling_factor]))
                ),
            title=title_x
            )

    points = alt.Chart(results, title=title).mark_circle(size=60).encode(
        x = X,
        y=alt.Y(metric, title=f"E({metric})", scale=alt.Scale(domain=(-1,2))),
        color=color,
    )

    line = alt.Chart(results).mark_line(color="red").encode(
        x = X,  y=alt.Y(f"E({metric})_pred", scale=alt.Scale(domain=(-1,2)))
    )

    (points + line).save(output_folder_fig / (file_name + ".pdf"))

    print(slope, intercept)
    # Our Scaling Law looks different from Kaplan's because our loss is already a log loss
    scaling_law = f"$L(N) = {slope:.5f} * ln(N) + {intercept:.5f} = {intercept:.5f} + ln(\\frac{{{1}}}{{N^{{{slope:.5f}}}}})$" 
    print(scaling_law)
    with open(output_folder_tex / (file_name + ".tex"), "w") as f:
        f.write(scaling_law)

if __name__ == "__main__":
    calc_scaling_law(
        scaling_factor="parameters",
        metric="MASE",
        file_name="scale_law_size_mase",
        title="Scaling Law with Parameter Count N",
        title_x="#Parameters (non-embedding)"
    )
    calc_scaling_law(
        scaling_factor="parameters",
        metric="RMSE",
        file_name="scale_law_size_rmse",
        title="Scaling Law with Parameter Count N",
        title_x="#Parameters (non-embedding)"
    )

    calc_scaling_law(
        scaling_factor="budget",
        metric="MASE",
        file_name="scale_law_duration_mase",
        title="Scaling Law with Training Duration D",
        title_x="#Training Samples"
    )

    # Take with grain of salt most tokens by long contexts might be padding token
    calc_scaling_law(
        scaling_factor="training_tokens",
        metric="MASE",
        file_name="scale_law_token_mase",
        title="Scaling Law with Training Token D",
        title_x="#Training Token"
    )

    calc_scaling_law(
        scaling_factor="compute",
        metric="MASE",
        file_name="scale_law_compute_mase",
        title="Scaling Law with Training Compute C",
        title_x="Compute in FLOPs"
    )

    calc_scaling_law(
        scaling_factor="parameters",
        metric="MASE",
        hp_adjusted=True,
        file_name="scale_law_size_mase_hp_adjusted",
        title="Scaling Law with Parameter Count N, when only considering best HP Config for each Size N and Train Duration D",
        title_x="#Parameters (non-embedding)",
        color=alt.Color("budget", title="#Train Samples")
    )