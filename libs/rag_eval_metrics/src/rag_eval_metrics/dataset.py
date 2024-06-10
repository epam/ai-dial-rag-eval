from datetime import datetime
from typing import Optional, Union

import fsspec
import pandas as pd
from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass


@dataclass
class DatasetMetadata:
    created_at: Optional[datetime]
    sources: list["Dataset"]
    tools: dict[str, str]
    file_info: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    statistics: dict = Field(default_factory=dict)

    @staticmethod
    def read_metadata(metadata_path: str) -> "DatasetMetadata":
        fs, _ = fsspec.url_to_fs(metadata_path)
        return TypeAdapter(DatasetMetadata).validate_json(fs.read_bytes(metadata_path))

    @staticmethod
    def create_missing_metadata(data_path: str) -> "DatasetMetadata":
        fs, _ = fsspec.url_to_fs(data_path)
        file_info = fs.info(data_path)
        return DatasetMetadata(None, [], {}, file_info, {}, {})


@dataclass
class Dataset:
    metadata_path: str
    data_path: str
    metadata: DatasetMetadata

    @staticmethod
    def handle_path(path: str) -> tuple[str, str]:
        if path.endswith(".parquet.metadata.json"):
            return path, path[: -len(".metadata.json")]
        if path.endswith(".parquet"):
            return path + ".metadata.json", path
        raise ValueError(f"Invalid path: {path}")

    @staticmethod
    def from_metadata(path: str):
        metadata_path, data_path = Dataset.handle_path(path)

        fs, _ = fsspec.url_to_fs(data_path)
        if fs.exists(metadata_path):
            metadata = DatasetMetadata.read_metadata(metadata_path)
        else:
            metadata = DatasetMetadata.create_missing_metadata(data_path)

        return Dataset(metadata_path, data_path, metadata)

    def read_dataframe(self, *args, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(self.data_path, *args, **kwargs)

    @staticmethod
    def write_dataframe(
        df: pd.DataFrame,
        path: str,
        sources: Optional[list["Dataset"]] = None,
        tools: Optional[dict[str, str]] = None,
        metrics: Optional[dict[str, float]] = None,
        statistics: Optional[dict[str, float]] = None,
        *args,
        **kwargs,
    ):
        sources = sources or []
        tools = tools or {}
        metrics = metrics or {}
        statistics = statistics or {}
        metadata_path, data_path = Dataset.handle_path(path)

        fs, _ = fsspec.url_to_fs(data_path)
        df.to_parquet(data_path, *args, **kwargs)
        file_info: dict = fs.info(data_path)

        metadata = DatasetMetadata(
            datetime.now(),
            sources,
            tools,
            file_info,
            metrics,
            statistics,
        )
        fs.write_bytes(
            metadata_path, TypeAdapter(DatasetMetadata).dump_json(metadata, indent=2)
        )

        return Dataset(metadata_path, data_path, metadata)


def source_dataset(dataset: Union[str, Dataset]):
    if isinstance(dataset, str):
        return Dataset.from_metadata(dataset)
    return dataset
