import os
import posixpath
from dataclasses import dataclass

import fsspec
import pandas as pd
import streamlit as st

# TODO : Make fs and dir paths configurable
fs = fsspec.filesystem("file")


data_root_path = os.environ.get("DATA_ROOT_PATH", "./data")
# data_root_path = os.environ.get("DATA_ROOT_PATH", "../../data")

data_dir = fs.info(data_root_path, detail=False)["name"]
ground_truth_dir = posixpath.join(data_dir, "ground_truth")
answers_dir = posixpath.join(data_dir, "answers")
evaluation_results_dir = posixpath.join(data_dir, "evaluation_results")
questions_dir = posixpath.join(data_dir, "questions")


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


def get_questions_datasets_list() -> list[DatasetPath]:
    return list_data_files(questions_dir)


@st.cache_data
def read_dataset(dataset_path: DatasetPath):
    return pd.read_parquet(dataset_path.path)


@st.cache_data
def read_ground_truth(dataset_path: DatasetPath):
    return read_dataset(dataset_path)


@st.cache_data
def read_answers(dataset_path: DatasetPath):
    return read_dataset(dataset_path)


@st.cache_data
def read_questions(dataset_path: DatasetPath):
    return read_dataset(dataset_path)


def write_dataset(
    parent_dir: str, name: str, data: pd.DataFrame, rewrite: bool = False
) -> DatasetPath:
    fs.makedirs(parent_dir, exist_ok=True)
    if not name.endswith(".parquet"):
        name = f"{name}.parquet"
    dataset_path = DatasetPath.create_as_relative(
        posixpath.join(parent_dir, name), parent_dir
    )
    if not rewrite and fs.exists(dataset_path.path):
        raise FileExistsError(f"File {dataset_path} already exists")

    data.to_parquet(dataset_path.path)
    return dataset_path


def write_evaluation_results(name: str, data: pd.DataFrame) -> DatasetPath:
    return write_dataset(evaluation_results_dir, name, data, rewrite=True)


def write_questions(
    name: str, data: pd.DataFrame, rewrite: bool = False
) -> DatasetPath:
    return write_dataset(questions_dir, name, data, rewrite)


def write_answers(name: str, data: pd.DataFrame, rewrite: bool = False) -> DatasetPath:
    return write_dataset(answers_dir, name, data, rewrite)


def write_ground_truth(
    name: str, data: pd.DataFrame, rewrite: bool = False
) -> DatasetPath:
    return write_dataset(ground_truth_dir, name, data, rewrite)
