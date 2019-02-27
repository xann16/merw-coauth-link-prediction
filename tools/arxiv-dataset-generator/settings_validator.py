import dsgen_utils as util


def validate_settings(settings):
    if "category" not in settings:
        raise BaseException("JSON settings file must contain 'category' " +
                            "field.")
    if "datasets" not in settings:
        raise BaseException("JSON settings file must contain 'datasets' " +
                            "field.")
    for dataset in settings["datasets"]:
        validate_dataset_entry(dataset)


def validate_dataset_entry(entry):
    if "name" not in entry:
        raise BaseException("JSON settings dataset entry must contain " +
                            "'name' field.")

    validate_split_method(entry)
    validate_period_field(entry, "JSON settings dataset entry")

    if "disable" not in entry:
        entry["disable"] = False
    else:
        bool(entry["disable"])

    if "disable_overwrite" not in entry:
        entry["disable_overwrite"] = False
    else:
        bool(entry["disable_overwrite"])

    if "disable_maxcc_extraction" not in entry:
        entry["disable_maxcc_extraction"] = False
    else:
        bool(entry["disable_maxcc_extraction"])


def validate_random_split(entry):
    if "test_perc" not in entry:
        raise BaseException("JSON settings dataset entry must contain " +
                            "'test_perc' parameter field for 'random' " +
                            "split method.")

    if "series_count" not in entry:
        entry["series_count"] = 1
    else:
        int(entry["series_count"])


def validate_k_cross_random_split(entry):
    if "k_subset_count" not in entry:
        entry["k_subset_count"] = 10
    else:
        k = int(entry["k_subset_count"])
        if k < 2:
            k = 2


SPLIT_METHOD_VALIDATORS = {"random": validate_random_split,
                           "k-cross-random": validate_k_cross_random_split}


def validate_split_method(entry):
    if "split_method" not in entry:
        raise BaseException("JSON settings dataset entry must contain " +
                            "'split_method' field.")
    sm = entry["split_method"]
    if sm not in SPLIT_METHOD_VALIDATORS:
        raise BaseException("JSON settings dataset entry contains " +
                            "unsupported 'split_method' ({}).".format(sm))
    SPLIT_METHOD_VALIDATORS[sm](entry)


def validate_period_field(entry, context):
    if "period" not in entry:
        raise BaseException("{} must contain 'period' field".format(context))
    period = entry["period"]
    if "from" not in period:
        raise BaseException("Period data must contain 'from' field")
    if "to" not in period:
        raise BaseException("Period data must contain 'to' field")

    t_from, t_to = util.parse_period(period)
    if t_from >= t_to:
        raise BaseException("Provided period bounds are invalid." +
                            " 'to' must be later than 'from'.")
