from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class FeatureDataset:
    def __init__(self, manifest: pd.DataFrame, root: str | Path):
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise RuntimeError("torch is required for training") from exc

        class _Dataset(Dataset):
            def __init__(self, rows: pd.DataFrame, base_dir: Path):
                self.rows = rows.reset_index(drop=True)
                self.base_dir = base_dir

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                row = self.rows.iloc[index]
                feature_path = Path(row["feature_path"])
                if not feature_path.is_absolute():
                    feature_path = self.base_dir / feature_path
                with np.load(feature_path) as data:
                    feature = data["feature"].astype(np.float32)
                tensor = torch.from_numpy(feature).unsqueeze(0)
                label = torch.tensor(int(row["label_id"]), dtype=torch.long)
                return tensor, label

        self.dataset = _Dataset(manifest, Path(root))

    def unwrap(self):
        return self.dataset
