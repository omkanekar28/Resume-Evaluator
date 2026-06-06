import argparse
import json
import re
from typing import Optional, Tuple

import pandas as pd


class FitRowReducer:
    """
    Reduces rows in a dataset by a specified percentage for a selected fit category.
    """

    FIT_CATEGORIES = {"strong", "moderate", "weak"}
    JSON_CODEBLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    FIT_PATTERN = re.compile(r"\b(strong|moderate|weak)\s+fit\b", re.IGNORECASE)

    def __init__(
        self,
        input_filepath: str,
        output_filepath: Optional[str],
        category: str,
        keep_percent: float,
        response_column: str = "Response",
        random_seed: Optional[int] = None,
    ) -> None:
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath or input_filepath
        self.category = category.lower()
        self.keep_percent = keep_percent
        self.response_column = response_column
        self.random_seed = random_seed

        if self.category not in self.FIT_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {sorted(self.FIT_CATEGORIES)}")

        if not 0 <= self.keep_percent <= 100:
            raise ValueError("keep_percent must be between 0 and 100")

    def load_dataframe(self) -> pd.DataFrame:
        if self.input_filepath.lower().endswith(".csv"):
            return pd.read_csv(self.input_filepath)
        return pd.read_excel(self.input_filepath)

    def save_dataframe(self, df: pd.DataFrame) -> None:
        if self.output_filepath.lower().endswith(".csv"):
            df.to_csv(self.output_filepath, index=False)
        else:
            df.to_excel(self.output_filepath, index=False)

    def parse_response_fit(self, response: str) -> str:
        if not isinstance(response, str):
            return "unknown"

        response_text = response.strip()
        fit_candidate = self._extract_fit_from_json(response_text)

        if fit_candidate:
            return fit_candidate

        fit_candidate = self._extract_fit_from_text(response_text)
        return fit_candidate or "unknown"

    def _extract_fit_from_json(self, response_text: str) -> Optional[str]:
        json_text = self._extract_json_block(response_text)
        if not json_text:
            return None

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return None

        final_assessment = parsed.get("final_assessment") or parsed.get("summary")
        if isinstance(final_assessment, str):
            return self._extract_fit_from_text(final_assessment)
        return None

    def _extract_json_block(self, response_text: str) -> Optional[str]:
        match = self.JSON_CODEBLOCK_PATTERN.search(response_text)
        if match:
            return match.group(1)

        # Fallback: capture the first curly-brace JSON object in the text.
        curly_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        return curly_match.group(1) if curly_match else None

    def _extract_fit_from_text(self, text: str) -> Optional[str]:
        match = self.FIT_PATTERN.search(text)
        if match:
            return match.group(1).lower()
        return None

    def reduce(self) -> Tuple[pd.DataFrame, dict]:
        df = self.load_dataframe()
        if self.response_column not in df.columns:
            raise KeyError(f"Response column '{self.response_column}' not found in input file.")

        df = df.copy()
        df["_fit_category"] = df[self.response_column].apply(self.parse_response_fit)

        total_counts = df["_fit_category"].value_counts(dropna=False).to_dict()
        candidate_rows = df[df["_fit_category"] == self.category]
        keep_count = int(round(len(candidate_rows) * self.keep_percent / 100))

        if self.random_seed is not None:
            candidate_rows = candidate_rows.sample(frac=1, random_state=self.random_seed)
        else:
            candidate_rows = candidate_rows.sample(frac=1)

        keep_rows = candidate_rows.iloc[:keep_count]
        rejected_rows = candidate_rows.iloc[keep_count:]

        preserved_rows = df[df["_fit_category"] != self.category]
        result_df = pd.concat([preserved_rows, keep_rows], ignore_index=True)

        stats = {
            "total_rows": len(df),
            "category": self.category,
            "original_category_count": len(candidate_rows),
            "keep_percent": self.keep_percent,
            "keep_count": len(keep_rows),
            "removed_count": len(rejected_rows),
            "category_counts": total_counts,
        }

        result_df.drop(columns=["_fit_category"], inplace=True)
        return result_df, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce rows for a given fit category in a JD/resume dataset."
    )
    parser.add_argument("input_filepath", help="Input dataset file path (.xlsx or .csv)")
    parser.add_argument(
        "--output_filepath",
        help="Output file path. If omitted, the input file is overwritten.",
        default=None,
    )
    parser.add_argument(
        "--category",
        choices=["strong", "moderate", "weak"],
        required=True,
        help="Fit category to reduce."
    )
    parser.add_argument(
        "--keep_percent",
        type=float,
        default=50.0,
        help="Percentage of selected category rows to keep (0-100)."
    )
    parser.add_argument(
        "--response_column",
        default="Response",
        help="Name of the column containing the model response text."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reducer = FitRowReducer(
        input_filepath=args.input_filepath,
        output_filepath=args.output_filepath,
        category=args.category,
        keep_percent=args.keep_percent,
        response_column=args.response_column,
        random_seed=args.random_seed,
    )

    try:
        output_df, stats = reducer.reduce()
        reducer.save_dataframe(output_df)

        print("Fit row reduction completed successfully.")
        print(f"Input file: {args.input_filepath}")
        print(f"Output file: {reducer.output_filepath}")
        print(f"Category reduced: {stats['category']}")
        print(f"Original category count: {stats['original_category_count']}")
        print(f"Rows kept: {stats['keep_count']}")
        print(f"Rows removed: {stats['removed_count']}")
        print(f"Category counts in input: {stats['category_counts']}")
    except Exception as exc:
        print(f"Error: {exc}")
