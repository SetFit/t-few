from typing import Dict, Tuple
from datasets import load_dataset, Dataset

import numpy as np

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))

def sentence_pairs_generation(sentences, labels, pairs):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    num_classes = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in num_classes]

    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        label = labels[first_idx]
        second_idx = first_idx
        while second_idx == first_idx:
            second_idx = np.random.choice(idx[np.where(num_classes == label)[0][0]])
        positive_sentence = sentences[second_idx]
        # Prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[current_sentence, positive_sentence], label=1.0))

        negative_idx = np.where(labels != label)[0]
        negative_sentence = sentences[np.random.choice(negative_idx)]
        # Prepare a negative pair of sentences and update our lists
        pairs.append(InputExample(texts=[current_sentence, negative_sentence], label=0.0))
    # Return a 2-tuple of our sentence pairs and labels
    return pairs


def create_pairs_dataset(dataset, num_iterations: int=20) -> Dict[str, Dataset]:
    # Randomly select `num_pairs` pairs from the test split,
    # and assign them Yes (in-class) / No (out-of-class) labels.

    x_train = dataset["text"]
    y_train = dataset["label"]
    examples = []

    for _ in range(num_iterations):
        pair_examples = sentence_pairs_generation(
            np.array(x_train), np.array(y_train), examples
        )

    # Construct Dataset from examples
    text_a_list, text_b_list, labels = [], [], []
    for example in pair_examples:
        text_a_list.append(example.texts[0])
        text_b_list.append(example.texts[1])
        labels.append(example.label)

    pairs_dataset = Dataset.from_dict(dict(text_a=text_a_list, text_b=text_b_list, label=labels))
    return pairs_dataset

def generate_jsonl_test_pairs(ds_name: str, num_samples: int=300) -> None:
    ds = load_dataset(ds_name)["test"].shuffle().select(range(num_samples))
    pair_datasets = create_pairs_dataset(ds)
    pair_datasets.to_json(f"pair_datasets/{ds_name}/test.jsonl")


def main():
    generate_jsonl_test_pairs("SetFit/sst2")


if __name__ == "__main__":
    main()
