import sqlite3
import pandas as pd


DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/similarity/similarity.db"
)


def get_best_results():

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            dataset,
            id,
            accuracy,
            auroc
        FROM runs
        WHERE status = 'DONE'
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    results = []

    for dataset, group in df.groupby("dataset"):

        result = {
            "dataset": dataset,
        }

        # ==================================================
        # Best Accuracy
        # ==================================================

        valid = group.dropna(subset=["accuracy"])

        if not valid.empty:

            row = valid.loc[valid["accuracy"].idxmax()]

            result["best_accuracy_id"] = int(row["id"])
            result["best_accuracy"] = row["accuracy"]

        else:

            result["best_accuracy_id"] = None
            result["best_accuracy"] = None


        # ==================================================
        # Best AUROC
        # ==================================================

        valid = group.dropna(subset=["auroc"])

        if not valid.empty:

            row = valid.loc[valid["auroc"].idxmax()]

            result["best_auroc_id"] = int(row["id"])
            result["best_auroc"] = row["auroc"]

        else:

            result["best_auroc_id"] = None
            result["best_auroc"] = None


        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    best_results = get_best_results()

    # ========================================================
    # Average across datasets
    # ========================================================

    average_row = pd.DataFrame([{
        "dataset": "Average",

        "best_accuracy_id": None,
        "best_accuracy": best_results["best_accuracy"].mean(),

        "best_auroc_id": None,
        "best_auroc": best_results["best_auroc"].mean(),
    }])

    best_results = pd.concat(
        [
            best_results,
            average_row
        ],
        ignore_index=True
    )

    # ========================================================
    # Print
    # ========================================================

    print("\nBest results per dataset:")

    print(
        best_results.to_string(index=False)
    )

    # ========================================================
    # Save
    # ========================================================

    output = (
        "/data/horse/ws/juha972b-AION-BERT-Chronos/"
        "BERTi/Results/Finetuning/Similarity/"
        "best_results.csv"
    )

    best_results.to_csv(
        output,
        index=False
    )

    print(f"\nSaved to {output}")