import json
import sys
from os import path
import process_dataset as impl
import settings_validator as validator
import dsgen_utils as util


def load_settings(settings_path):
    if len(settings_path) < 5 or not settings_path.endswith(".json"):
        settings_path += ".json"
    if not (path.exists(settings_path) and path.isfile(settings_path)):
        raise BaseException('Settings json file not found. Expected: {}'
                            .format(path.abspath(settings_path)))
    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)
        return settings


def prepare_dirs(base_path, settings):
    util.create_dir(base_path, ".arxiv-cache")
    util.create_dir(base_path, path.join(".arxiv-cache", settings["category"]))
    util.create_dir(base_path, settings["category"])
    for dataset in settings["datasets"]:
        util.create_dir(base_path,
                        path.join(settings["category"], dataset["name"]),
                        overwrite=not dataset["disable_overwrite"])


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Invalid number of command line arguments.')
    else:
        try:
            settings_path = sys.argv[1]
            base_dir = "."
            if len(sys.argv) == 3:
                base_dir = sys.argv[2]

            settings = load_settings(settings_path)
            validator.validate_settings(settings)
            prepare_dirs(base_dir, settings)

            index = 1
            total = len(settings["datasets"])
            for dataset in settings["datasets"]:
                try:
                    print('[{}/{}] Processing dataset: {}...'
                          .format(index, total, dataset["name"]))
                    impl.process_dataset(settings["category"],
                                         dataset,
                                         path.abspath(base_dir))
                except BaseException as e:
                    print('Processing of {} aborted: {}'
                          .format(dataset["name"], e))
                finally:
                    index += 1

        except BaseException as e:
            print("arxiv_dsgen:", e)
