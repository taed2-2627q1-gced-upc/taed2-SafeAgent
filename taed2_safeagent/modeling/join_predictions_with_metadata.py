from pathlib import Path

import pandas as pd

from taed2_safeagent.config import PROJ_ROOT, MODEL_NAMES


def main():
    # Read source metadata, located at file "provenance.jsonl"
    source_data_path = PROJ_ROOT / "data/interim/shell_safety/provenance.jsonl"
    source_df = pd.read_json(source_data_path, lines=True)

    # Normalize the "source" column
    source_df_extra = pd.json_normalize(source_df["sources"].str[0])  # type: ignore
    source_df = pd.concat([source_df.drop(columns=["sources"]), source_df_extra], axis=1)

    # Data with all features and labels, located at file "clean.jsonl"
    clean_data_path = PROJ_ROOT / "data/interim/shell_safety/clean.jsonl"
    clean_df = pd.read_json(clean_data_path, lines=True)

    # For each model predictions, join with metadata
    for model_name in MODEL_NAMES:
        # Get model predictions and join them with source metadata
        predictions_path = PROJ_ROOT / Path(
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

        # Save joined dataset
        joined_df_path = PROJ_ROOT / Path(
            f"models/encoders/{model_name}/predictions_joined_with_metadata.jsonl"
        )
        joined_df.to_json(joined_df_path, orient="records", lines=True)

        print(
            f"\nPredictions from {model_name}, located at {predictions_path}, "
            f"were merged with source metadata on {joined_df_path}"
        )


if __name__ == "__main__":
    main()
