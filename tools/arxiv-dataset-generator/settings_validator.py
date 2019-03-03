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

    validate_period_field(entry, "JSON settings dataset entry")
    validate_split_method(entry)

    if "disable" not in entry:
        entry["disable"] = False
    else:
        bool(entry["disable"])

    if "disable_overwrite" not in entry:
        entry["disable_overwrite"] = False
    else:
        bool(entry["disable_overwrite"])


def validate_random_split(entry):
    if "test_perc" not in entry:
        entry["test_perc"] = 10
    else:
        int(entry["test_perc"])

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


def validate_chrono_perc_split(entry):
    if "test_perc" not in entry:
        entry["test_perc"] = 10
    else:
        int(entry["test_perc"])


def validate_chrono_from_split(entry):
    if "test_from" not in entry:
        raise BaseException("JSON settings dataset entry must contain " +
                            "'test_from' field for 'chrono-from' split" +
                            " method.")
    else:
        util.parse_datetime(entry["test_from"])


def validate_chrono_from_date_in_period(entry):
    t_from, t_to = util.parse_period(entry["period"])
    t_split = util.parse_datetime(entry["test_from"])
    if t_split < t_from or t_split > t_to:
        raise BaseException("In 'chrono-from' split method date provided in" +
                            " 'test_from' field must be within 'period'")


SPLIT_METHOD_VALIDATORS = {"random": validate_random_split,
                           "k-cross-random": validate_k_cross_random_split,
                           "chrono-perc": validate_chrono_perc_split,
                           "chrono-from": validate_chrono_from_split}


def validate_split_method(entry):
    if "split_method" not in entry:
        raise BaseException("JSON settings dataset entry must contain " +
                            "'split_method' field.")
    sm = entry["split_method"]
    if sm not in SPLIT_METHOD_VALIDATORS:
        raise BaseException("JSON settings dataset entry contains " +
                            "unsupported 'split_method' ({}).".format(sm))
    SPLIT_METHOD_VALIDATORS[sm](entry)

    if sm == "chrono-from":
        validate_chrono_from_date_in_period(entry)


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
