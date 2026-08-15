from __future__ import annotations

from pathlib import Path

from config import SETTINGS
from microstructure_model import train_from_sqlite


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def main():
    Path(SETTINGS.model_dir).mkdir(parents=True, exist_ok=True)
    for symbol in SETTINGS.symbols:
        out = str(Path(SETTINGS.model_dir) / f"micro_{safe_name(symbol)}.joblib")
        try:
            model, rows = train_from_sqlite(SETTINGS.db_path, symbol, out)
        except ValueError as e:
            print(f"{symbol}: not ready — {e}")
            continue

        print(f"\n{symbol} | labeled rows={rows}")
        for h in model.horizons_seconds:
            print(
                f"  {h:>3}s  MAE={model.validation_mae[h]:.6f} "
                f"directional_accuracy={model.directional_accuracy[h]:.3f}"
            )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
