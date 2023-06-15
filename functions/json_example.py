import sys, os
import json
import pprint
import logging


def read_control_json(json_file):
    """
    Reads in json file to dictionary, also defining paths.

    Parameters
    ----------
    json file: file path
        complete path to json file

    Returns
    -------
    settings : dict
        dictionary of settings
    """
    f = open(json_file)
    settings = json.load(f)

    settings = set_paths(settings, "fulldom_data", file_expected=True)

    return settings


def set_paths(settings, path_type, file_expected=False):
    """
    Update all paths in settings dictionary

    Parameters
    ----------
    settings: dict
        dictionary of run settings read from json
    path_type: str
        path type to be updated, example "basin_data"
    file_expected: bool
        check if the files are expected, such as needed input data files.

    Returns
    -------
    settings : dict
        update settings dictionary

    """

    full_path = path_type + "_path"

    for key in settings[path_type]:

        if settings[path_type][key]["path"] == full_path:
            settings[path_type][key]["loc"] = (
                settings["paths"][full_path] + settings[path_type][key]["name"]
            )

        else:
            settings[path_type][key]["loc"] = (
                settings[path_type][key]["path"] + settings[path_type][key]["name"]
            )

        if file_expected and not os.path.exists(settings[path_type][key]["loc"]):
            logging.error(
                f"Expected file {settings[path_type][key]['loc']} not found. Check path and file location"
            )

    return settings


if __name__ == "__main__":
    if len(sys.argv) > 1:
        control_json_file = sys.argv[1]
    else:
        control_json_file = "../test_cases/taylorpark/control.taylorpark.json"

    settings = read_control_json(control_json_file)
    pprint.pprint(settings)
