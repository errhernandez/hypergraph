
import re

def write_node_features(features: list[str]) -> str:

    log_text = "---\n"
    log_text += "##        Node Features: \n"
    log_text += "---\n"
    
    for feature in features:

        if re.match("^gro|^per", feature):
    
           n_features = " 25 (one-hot encoded)\n" # 7 period, 18 group = 25

        else:

           n_features = "       1 float\n"

        log_text += "- " + feature + "    " + n_features

    return log_text

def write_parameters(head: str, parameters: dict,
                      units: str = "Angstrom") -> str:

    log_text = "---\n"
    log_text += "##    parameters for: " + head + "\n"
    log_text += "---\n"
    log_text += "- nFeatures: " + repr(parameters["n_features"]) + "\n"
    log_text += "- r_min: " + repr(parameters["x_min"]) + " " + units + "\n"
    log_text += "- r_max: " + repr(parameters["x_max"]) + " " + units + "\n"
    log_text += "- sigma: " + repr(parameters["sigma"]) + " " + units + "\n"
    if parameters["norm"]:
        log_text += "- Normalised: " + "True\n"
    log_text += "---\n"

    return log_text

