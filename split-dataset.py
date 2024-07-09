import json
import datasets
from datasets import load_dataset
import os


_CITATION = """"""

_DESCRIPTION = """"""

_HOMEPAGE = ""

_LICENSE = "CC BY-NC-ND 4.0"



class Text2Chart31(datasets.GeneratorBasedBuilder):
    """Text2Chart31 Corpus dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="Text2Chart31"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "description": datasets.Value("string"),
                "code": datasets.Value("string"),
                "csv-address": datasets.Value("string"),
                "csv-name": datasets.Value("string"),
                "data-table": datasets.Value("string"),
                "data-type": datasets.Value("string"),
                "plot-category": datasets.Value("string"),
                "plot-type": datasets.Value("string"),
                "reasoning-step": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = "."
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": (path + "/prepare-data/", "Text2Chart-31-train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": (path + "/prepare-data/", "Text2Chart-31-test.json"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        path, fname = filepath
        bio = os.path.join(path, fname)
        with open(bio, "r") as f:
            data = json.load(f)
        for example in data:
            yield example["id"], example
