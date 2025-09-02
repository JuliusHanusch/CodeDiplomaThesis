from constants import *
from scaling_laws import load_data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import plotly.express as px
from typing import List


metric = "MASE"


def preprocess(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, List[str], dict[str, list]]:
    dims = [c for c in features if df[c].nunique(dropna=False) > 1]
    ticktxt = {}
    for c in dims:
        if df[c].dtype == "object":
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            ticktxt[c] = le.classes_.tolist()
    num = df[dims].select_dtypes("number").columns
    # if num.any():
    #     df[num] = MinMaxScaler().fit_transform(df[num])
    return df, dims, ticktxt

def order_and_rho(df: pd.DataFrame, dims: List[str]) -> tuple[List[str], dict[str, float]]:
    rho = df[dims + [metric]].corr("spearman")[metric].abs().drop(metric)
    return rho.sort_values(ascending=False).index.tolist(), rho.to_dict()

def build(df, dims, ticktxt, rho):
    # label_map = {d: f"{d.replace('_',' ')} (ρ={rho[d]:.2f})" for d in dims}
    fig = px.parallel_coordinates(
        df,
        dimensions=dims,
        color=metric,
        color_continuous_scale="Cividis",
        # reversescale=True,
        labels=dims,
    )# TODO.update_traces(opacity=0.4)

    parcoords = fig.data[0]  # first (and only) trace is a parcoords object
    for i, d in enumerate(dims):
        dim = parcoords.dimensions[i]
        dim.label = f"{d.replace('_',' ')} (ρ={rho[d]:.2f})"
        if d in ticktxt:
            dim.tickvals = list(range(len(ticktxt[d])))
            dim.ticktext = ticktxt[d]
        else:
            dim.tickformat = ".2f"
        # dim.label = "Your Label"

    fig.update_layout(
        font_size=10,
        coloraxis_colorbar=dict(title=metric, len=0.75),
        margin=dict(l=40, r=40, t=20, b=20),
    ).add_annotation(
        xref="paper", yref="paper", x=0, y=-0.15, showarrow=False,
        font_size=9,
        text=("Figure 1: Parallel-coordinates visualization of hyper-parameter "
              "configurations; darker lines indicate lower MASE."),
    )
    return fig


if __name__ == "__main__":
    results = load_data()
    results_hp = results['config'].apply(json.loads).apply(pd.Series)
    results_hp = results_hp.drop(["seed"], axis="columns")
    features = list(results_hp.columns)
    results = pd.concat([results.drop(columns='config'), results_hp], axis=1)
    results = results.drop(["seed"], axis="columns")
    print(results)

    df, dims, ticktxt = preprocess(results, features)
    dims, rho = order_and_rho(df, dims)
    fig = build(df, dims, ticktxt, rho)

    # fig = px.parallel_coordinates(
    #     results,
    #     dimensions=features,
    #     color=metric,
    #     color_continuous_scale='Viridis'
    # )
    fig.write_image(output_folder_fig / "hp_coordinates.pdf", width=1200, height=1200, scale=2, format="pdf")
    fig.write_html(output_folder_fig / "hp_coordinates.html", include_plotlyjs="cdn")
    # fig.write_html(OUT_HTML, include_plotlyjs="cdn")

    sys.exit()
    # Normalize features (optional)
    results_scaled = results.copy()
    features_num = list(results_scaled[features].select_dtypes(include='number').columns)



    # TODO Include only numeric features + Switch to log scale if necessary 
    results_scaled[features_num] = MinMaxScaler().fit_transform(results_scaled[features_num])

    # Melt to long format
    results_long = results_scaled[features+["id", metric]].melt(id_vars=['id', metric], var_name='feature', value_name='value')

    # Create parallel coordinates plot
    chart = alt.Chart(results_long).mark_line(opacity=0.6).encode(
        x='feature:N',
        y='value:Q',
        detail='id:N',
        color=f'{metric}'  
    ).properties(width=600, height=300)

    chart.save(output_folder_fig / "hp_coordinates.pdf")