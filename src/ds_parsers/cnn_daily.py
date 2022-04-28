import os
import json
from tqdm import tqdm

DEFAULT_CNN_DAILY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cnn_daily")


def __parse_story(file_path: str):
    Xi = ""
    yi = ""

    to_y = False

    for line in open(file_path, 'r'):
        line = line.strip()
        if len(line) == 0:
            continue
        if to_y:
            if "@highlight" not in line:
                yi += line + " . "
        else:
            Xi += line + " "
    return Xi, yi


def __cnn_daily_iterator(path: str):
    for file in os.listdir(path):
        ffile = os.path.join(path, file)
        if os.path.isdir(ffile):
            for subfile in __cnn_daily_iterator(ffile):
                yield subfile
        elif os.path.isfile(ffile) and ffile.endswith(".story"):
            yield os.path.join(path, ffile)


def parse_cnn_daily(path: str = DEFAULT_CNN_DAILY_PATH):
    X = []
    y = []

    print("Counting files in dataset: ", end="")
    n_files = sum([1 for _ in __cnn_daily_iterator(DEFAULT_CNN_DAILY_PATH)])

    def _parse(p: str):
        for file in tqdm(__cnn_daily_iterator(path), total=n_files):
            X_cur, y_cur = __parse_story(file)
            X.append(X_cur)
            y.append(y_cur)

        # for file in os.listdir(p):
        #     if os.path.isdir(file):
        #         _parse(file)
        #     elif os.path.isfile(file) and file.endswith(".story"):
        #         X_cur, y_cur = __parse_story(file)
        #         X.append(X_cur)
        #         y.append(y_cur)

    _parse(path)

    return X, y


if __name__ == '__main__':
    # print(sum([1 for _ in __cnn_daily_iterator(DEFAULT_CNN_DAILY_PATH)]))
    X, y = parse_cnn_daily()
    with open(os.path.join(DEFAULT_CNN_DAILY_PATH, "parsed.json"), 'w') as parsed_f:
        json.dump({
            "X": X,
            "y": y,
        }, parsed_f)
