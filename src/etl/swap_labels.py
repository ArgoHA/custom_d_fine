"""
Swap old label with new one
"""

from pathlib import Path

from tqdm import tqdm


def swap_labels(labels_path, output_path, label_old, label_new):
    output_path.mkdir(exist_ok=True, parents=True)
    for label in tqdm(labels_path.iterdir(), total=len(list(labels_path.iterdir()))):
        if label.suffix != ".txt" or label.name == "labels.txt":
            continue
        with open(label, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line[0] == str(label_old):
                lines[i] = str(label_new) + lines[i][1:]

        with open(output_path / label.name, "w") as f:
            f.writelines(lines)


def main():
    labels_path = Path("/home/argo/Desktop/Projects/Veryfi/dt/data/dataset/labels")
    output_path = labels_path.parent / f"{labels_path.name}_swapped"
    label_old = 2
    label_new = 0

    swap_labels(labels_path, output_path, label_old, label_new)


if __name__ == "__main__":
    main()
