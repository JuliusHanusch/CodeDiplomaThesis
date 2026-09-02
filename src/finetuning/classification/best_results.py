import sqlite3
import json
import statistics
import csv


# ============================================================
# CONFIG
# ============================================================

DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/classification/classification.db"
)

OUTPUT = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/Results/Finetuning/Classification/bert_small_seeds.csv"
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
            config,
            dataset,

            model_path_42,
            model_path_43,
            model_path_44,
            model_path_45,
            model_path_46,

            accuracy_42,
            accuracy_43,
            accuracy_44,
            accuracy_45,
            accuracy_46

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

    # Group all runs by dataset
    datasets = {}

    for row in rows:
        datasets.setdefault(row["dataset"], []).append(row)

    csv_rows = []

    # Values used for the final averages
    dataset_mean_accuracies = []
    dataset_best_single_accuracies = []

    # ========================================================
    # EACH DATASET
    # ========================================================

    for dataset, dataset_rows in datasets.items():

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset}")
        print("=" * 100)

        # ====================================================
        # 1. CONFIG WITH HIGHEST MEAN OVER 5 SEEDS
        # ====================================================

        config_results = []

        for row in dataset_rows:

            accuracies = []

            for seed in SEEDS:

                value = row[f"accuracy_{seed}"]

                if value is not None:
                    accuracies.append(float(value))

            # We need all 5 seeds
            if len(accuracies) != 5:
                continue

            mean_accuracy = statistics.mean(accuracies)
            std_accuracy = statistics.stdev(accuracies)

            config_results.append({
                "row": row,
                "mean": mean_accuracy,
                "std": std_accuracy,
                "accuracies": accuracies,
            })

        if config_results:

            # Highest average accuracy
            best_config = max(
                config_results,
                key=lambda x: x["mean"]
            )

            row = best_config["row"]

            mean_accuracy = best_config["mean"]
            std_accuracy = best_config["std"]

            dataset_mean_accuracies.append(mean_accuracy)

            print("\nBEST CONFIG")
            print("-" * 100)
            print(f"Config ID   : {row['id']}")
            print(f"Config hash : {row['config_hash']}")
            print(f"Mean        : {mean_accuracy:.6f}")
            print(f"Std         : {std_accuracy:.6f}")
            print(
                f"Result      : "
                f"{mean_accuracy:.6f} ± {std_accuracy:.6f}"
            )

            print("\nSeeds:")

            for seed, accuracy in zip(
                SEEDS,
                best_config["accuracies"]
            ):
                print(f"  {seed}: {accuracy:.6f}")

            print("\nConfig:")

            try:
                parsed_config = json.loads(row["config"])
                print(json.dumps(parsed_config, indent=4))
            except (TypeError, json.JSONDecodeError):
                print(row["config"])

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_rows.append({
                "dataset": dataset,
                "result_type": "best_mean",
                "config_id": row["id"],
                "accuracy": mean_accuracy,
                "std": std_accuracy,
                "seed": "",
            })

        else:

            print("\nBEST CONFIG")
            print("-" * 100)
            print("No configuration has results for all 5 seeds.")

        # ====================================================
        # 2. HIGHEST SINGLE ACCURACY
        # ====================================================

        best_single = None

        for row in dataset_rows:

            for seed in SEEDS:

                value = row[f"accuracy_{seed}"]

                if value is None:
                    continue

                accuracy = float(value)

                if (
                    best_single is None
                    or accuracy > best_single["accuracy"]
                ):
                    best_single = {
                        "accuracy": accuracy,
                        "seed": seed,
                        "row": row,
                    }

        if best_single is not None:

            row = best_single["row"]

            best_single_accuracy = best_single["accuracy"]

            dataset_best_single_accuracies.append(
                best_single_accuracy
            )

            print("\nBEST SINGLE RESULT")
            print("-" * 100)
            print(f"Accuracy    : {best_single_accuracy:.6f}")
            print(f"Seed        : {best_single['seed']}")
            print(f"Config ID   : {row['id']}")
            print(f"Config hash : {row['config_hash']}")

            print("\nConfig:")

            try:
                parsed_config = json.loads(row["config"])
                print(json.dumps(parsed_config, indent=4))
            except (TypeError, json.JSONDecodeError):
                print(row["config"])

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_rows.append({
                "dataset": dataset,
                "result_type": "best_single",
                "config_id": row["id"],
                "accuracy": best_single_accuracy,
                "std": "",
                "seed": best_single["seed"],
            })

        else:

            print("\nBEST SINGLE RESULT")
            print("-" * 100)
            print("No accuracy results available.")

    # ========================================================
    # 3. AVERAGE OF DATASET MEANS
    # ========================================================

    if dataset_mean_accuracies:

        average_seed_mean = statistics.mean(
            dataset_mean_accuracies
        )

        print("\n" + "=" * 100)
        print("AVERAGE OF DATASET MEANS")
        print("=" * 100)

        print(
            f"Average of dataset mean accuracies: "
            f"{average_seed_mean:.6f}"
        )

        csv_rows.append({
            "dataset": "AVERAGE",
            "result_type": "average_seed_mean",
            "config_id": "",
            "accuracy": average_seed_mean,
            "std": "",
            "seed": "",
        })

    # ========================================================
    # 4. AVERAGE OF BEST SINGLE RESULTS
    # ========================================================

    if dataset_best_single_accuracies:

        average_best_single = statistics.mean(
            dataset_best_single_accuracies
        )

        print("\n" + "=" * 100)
        print("AVERAGE OF BEST SINGLE RESULTS")
        print("=" * 100)

        print(
            f"Average of best single accuracies: "
            f"{average_best_single:.6f}"
        )

        csv_rows.append({
            "dataset": "AVERAGE",
            "result_type": "average_best_single",
            "config_id": "",
            "accuracy": average_best_single,
            "std": "",
            "seed": "",
        })

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "dataset",
        "result_type",
        "config_id",
        "accuracy",
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