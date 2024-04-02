import posixpath
from dataclasses import dataclass

import fsspec
import pandas as pd
import streamlit as st

# TODO : Make fs and dir paths configurable
fs = fsspec.filesystem("file")

data_dir = fs.info("data", detail=False)["name"]
ground_truth_dir = posixpath.join(data_dir, "ground_truth")
answers_dir = posixpath.join(data_dir, "answers")
evaluation_results_dir = posixpath.join(data_dir, "evaluation_results")


@dataclass(frozen=True)
class DatasetPath:
    display_name: str
    path: str

    def __str__(self):
        return self.display_name

    @classmethod
    def create_as_relative(cls, path: str, base_dir: str) -> "DatasetPath":
        relative_path = (
            path[len(base_dir) :].lstrip("/") if path.startswith(base_dir) else path
        )
        return cls(display_name=relative_path, path=str(path))


@st.cache_data
def list_data_files(dir_path) -> list[DatasetPath]:
    pattern = str(posixpath.join(dir_path, "**/*.parquet"))
    return [DatasetPath.create_as_relative(path, dir_path) for path in fs.glob(pattern)]


def get_all_datasets_list() -> list[DatasetPath]:
    return list_data_files(data_dir)


def get_ground_truth_datasets_list() -> list[DatasetPath]:
    return list_data_files(ground_truth_dir)


def get_answers_datasets_list() -> list[DatasetPath]:
    return list_data_files(answers_dir)


def get_evaluation_results_datasets_list() -> list[DatasetPath]:
    return list_data_files(evaluation_results_dir)


@st.cache_data
def read_dataset(dataset_path: DatasetPath):
    return pd.read_parquet(dataset_path.path)


@st.cache_data
def read_ground_truth(dataset_path: DatasetPath):
    return read_dataset(dataset_path)


@st.cache_data
def read_answers(dataset_path: DatasetPath):
    return read_dataset(dataset_path)


def write_evaluation_results(name: str, data: pd.DataFrame) -> DatasetPath:
    fs.makedirs(evaluation_results_dir, exist_ok=True)
    if not name.endswith(".parquet"):
        name = f"{name}.parquet"
    dataset_path = DatasetPath.create_as_relative(
        posixpath.join(evaluation_results_dir, name), evaluation_results_dir
    )
    data.to_parquet(dataset_path.path)
    return dataset_path
