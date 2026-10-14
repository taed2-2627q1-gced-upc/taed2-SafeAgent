from pathlib import Path

import pandas as pd


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    print(f"Project root: {PROJECT_ROOT}")

    # Read clean dataset for command and command length, located at file "clean.jsonl"
    clean_data_path = PROJECT_ROOT / Path("data/interim/shell_safety/clean.jsonl")
    clean_df = pd.read_json(clean_data_path, lines=True)

    # Read source metadata, located at file "provenance.jsonl"
    source_data_path = PROJECT_ROOT / Path("data/interim/shell_safety/provenance.jsonl")
    source_df = pd.read_json(source_data_path, lines=True)

    # Normalize the "source" column
    source_df_extra = pd.json_normalize(source_df["sources"].str[0])  # type: ignore
    source_df = pd.concat([source_df.drop(columns=["sources"]), source_df_extra], axis=1)

    # Define model names
    models_names = [
        "codebert__command__2__a740d875",
        "codebert__command-context__2__a740d875",
        "modernbert__command__2__a740d875",
        "modernbert__command-context__2__a740d875",
    ]

    # For each model predictions, join with metadata
    for model_name in models_names:
        # Get model predictions and join them with source metadata
        predictions_path = PROJECT_ROOT / Path(
            f"models/encoders/{model_name}/validation_predictions.jsonl"
        )
        predictions_df = pd.read_json(predictions_path, lines=True)
        joined_df = pd.merge(predictions_df, source_df, how="inner", on="id", validate="1:1")
        joined_df = pd.merge(
            joined_df, clean_df[["id", "command"]], how="inner", on="id", validate="1:1"
        )

        # Assert that no registries were lost
        if len(joined_df) != len(predictions_df):
            raise ValueError(
                f"Error D1: {len(predictions_df) - len(joined_df)} registries "
                f"were lost during merge from model {model_name}."
            )

        # Save joined dataset into disk
        joined_df_path = PROJECT_ROOT / Path(
            f"models/encoders/{model_name}/predictions_joined_with_metadata.jsonl"
        )
        joined_df.to_json(joined_df_path, orient="records", lines=True)

        print(
            f"\nPredictions from {model_name}, located at {predictions_path}, "
            f"were merged with source metadata on {joined_df_path}"
        )


if __name__ == "__main__":
    main()
