from datetime import datetime


colors = {
    'light_gray': '#929292',
    'dark_gray': '#494949',
    'orange': '#FF7800'
}


def strp_datetime(dt_string, fmt="%d/%m/%y"):
    return datetime.strptime(dt_string, fmt)


def strp_datetime_year_first(dt_string):
    return strp_datetime(dt_string, fmt="%d/%m/%y")


def to_sep_str(lst, sep=','):
    return sep.join([str(e) for e in lst])


def to_time(pnt_nb, t_start, sample_interval):
    return t_start + (pnt_nb * sample_interval)


def check_is_odd(nb):
    nb = int(nb)
    if nb % 2:
        return nb
    else:
        raise ValueError("Expected odd number received: {}".format(nb))
