import sqlite3
import statistics
import csv


# ============================================================
# CONFIG
# ============================================================

DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/tser/tser.db"
)

OUTPUT = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/TSER/"
    "tser_small_seeds.csv"
)

SEEDS = [42, 43, 44, 45, 46]


# ============================================================
# LOAD DATABASE
# ============================================================

def load_results():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            id,
            config_hash,
            dataset,

            mae_42,
            mae_43,
            mae_44,
            mae_45,
            mae_46

        FROM runs
        WHERE dataset IS NOT NULL
    """).fetchall()

    conn.close()

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    rows = load_results()

    # Group runs by dataset
    datasets = {}

    for row in rows:
        datasets.setdefault(row["dataset"], []).append(row)

    csv_rows = []

    # Values for final averages
    dataset_mean_maes = []
    dataset_best_single_maes = []

    # ========================================================
    # EACH DATASET
    # ========================================================

    for dataset, dataset_rows in datasets.items():

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset}")
        print("=" * 100)

        # ====================================================
        # 1. CONFIG WITH LOWEST MEAN MAE OVER 5 SEEDS
        # ====================================================

        config_results = []

        for row in dataset_rows:

            maes = []

            for seed in SEEDS:

                value = row[f"mae_{seed}"]

                if value is not None:
                    maes.append(float(value))

            # Require all 5 seeds
            if len(maes) != 5:
                continue

            mean_mae = statistics.mean(maes)
            std_mae = statistics.stdev(maes)

            config_results.append({
                "row": row,
                "mean": mean_mae,
                "std": std_mae,
                "maes": maes,
            })

        if config_results:

            # LOWER MAE IS BETTER
            best_config = min(
                config_results,
                key=lambda x: x["mean"]
            )

            row = best_config["row"]

            mean_mae = best_config["mean"]
            std_mae = best_config["std"]

            dataset_mean_maes.append(mean_mae)

            print("\nBEST CONFIG")
            print("-" * 100)
            print(f"ID          : {row['id']}")
            print(f"Config hash : {row['config_hash']}")
            print(f"Mean MAE    : {mean_mae:.6f}")
            print(f"Std MAE     : {std_mae:.6f}")
            print(
                f"Result      : "
                f"{mean_mae:.6f} ± {std_mae:.6f}"
            )

            print("\nSeeds:")

            for seed, mae in zip(
                SEEDS,
                best_config["maes"]
            ):
                print(f"  {seed}: {mae:.6f}")

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_rows.append({
                "dataset": dataset,
                "result_type": "best_mean",
                "id": row["id"],
                "config_id": row["id"],
                "mae": mean_mae,
                "std": std_mae,
                "seed": "",
            })

        else:

            print("\nBEST CONFIG")
            print("-" * 100)
            print("No configuration has results for all 5 seeds.")

        # ====================================================
        # 2. HIGHEST SINGLE / LOWEST SINGLE MAE
        # ====================================================

        best_single = None

        for row in dataset_rows:

            for seed in SEEDS:

                value = row[f"mae_{seed}"]

                if value is None:
                    continue

                mae = float(value)

                # LOWER MAE IS BETTER
                if (
                    best_single is None
                    or mae < best_single["mae"]
                ):
                    best_single = {
                        "mae": mae,
                        "seed": seed,
                        "row": row,
                    }

        if best_single is not None:

            row = best_single["row"]

            best_single_mae = best_single["mae"]

            dataset_best_single_maes.append(
                best_single_mae
            )

            print("\nBEST SINGLE RESULT")
            print("-" * 100)
            print(f"ID          : {row['id']}")
            print(f"MAE         : {best_single_mae:.6f}")
            print(f"Seed        : {best_single['seed']}")
            print(f"Config hash : {row['config_hash']}")

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_rows.append({
                "dataset": dataset,
                "result_type": "best_single",
                "id": row["id"],
                "config_id": row["id"],
                "mae": best_single_mae,
                "std": "",
                "seed": best_single["seed"],
            })

        else:

            print("\nBEST SINGLE RESULT")
            print("-" * 100)
            print("No MAE results available.")

    # ========================================================
    # 3. AVERAGE OF THE DATASET BEST MEANS
    # ========================================================

    if dataset_mean_maes:

        average_seed_mean = statistics.mean(
            dataset_mean_maes
        )

        print("\n" + "=" * 100)
        print("AVERAGE OF DATASET BEST MEANS")
        print("=" * 100)

        print(
            f"Average of dataset best mean MAEs: "
            f"{average_seed_mean:.6f}"
        )

        csv_rows.append({
            "dataset": "AVERAGE",
            "result_type": "average_best_mean",
            "id": "",
            "config_id": "",
            "mae": average_seed_mean,
            "std": "",
            "seed": "",
        })

    # ========================================================
    # 4. AVERAGE OF THE DATASET BEST SINGLE RESULTS
    # ========================================================

    if dataset_best_single_maes:

        average_best_single = statistics.mean(
            dataset_best_single_maes
        )

        print("\n" + "=" * 100)
        print("AVERAGE OF BEST SINGLE RESULTS")
        print("=" * 100)

        print(
            f"Average of dataset best single MAEs: "
            f"{average_best_single:.6f}"
        )

        csv_rows.append({
            "dataset": "AVERAGE",
            "result_type": "average_best_single",
            "id": "",
            "config_id": "",
            "mae": average_best_single,
            "std": "",
            "seed": "",
        })

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "dataset",
        "result_type",
        "id",
        "config_id",
        "mae",
        "std",
        "seed",
    ]

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 100)
    print("CSV SAVED")
    print("=" * 100)
    print(OUTPUT)


if __name__ == "__main__":
    main()