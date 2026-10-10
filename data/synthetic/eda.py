
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "retail_data.csv"
OUTPUT_DIR = BASE_DIR / "eda_output"


def main():
    df = pd.read_csv(DATA_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Purchase rate by treatment
    purchase_rates = df.groupby("treatment")["purchase"].mean()

    ax = purchase_rates.rename(
        index={0: "No discount", 1: "Discount"}
    ).plot(
        kind="bar",
        color=["#64748b", "#16804e"],
        figsize=(7, 5),
        rot=0,
    )
    ax.set_title("Purchase Rate by Treatment")
    ax.set_ylabel("Purchase probability")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "purchase_rate.png", dpi=150)
    plt.close()

    # 2. Purchase rate by product category and treatment
    category_rates = (
        df.groupby(["category", "treatment"])["purchase"]
        .mean()
        .unstack()
        .rename(columns={0: "No discount", 1: "Discount"})
    )

    ax = category_rates.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Purchase Rate by Category")
    ax.set_ylabel("Purchase probability")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_purchase_rate.png", dpi=150)
    plt.close()

    # 3. Numeric summary
    summary = df[
        [
            "baseline_price",
            "customer_history",
            "discount_pct",
            "purchase",
            "true_effect",
        ]
    ].describe()

    summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    print("EDA completed.")
    print(f"Charts and summary saved to: {OUTPUT_DIR}")
    print("\nPurchase rates:")
    print(purchase_rates)
    print("\nCategory purchase rates:")
    print(category_rates)


if __name__ == "__main__":
    main()
