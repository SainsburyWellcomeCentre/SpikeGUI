from margrie_libs.margrie_libs.utils.utils_exceptions import UtilsPrintException


def dprint(input_str):
    """
    prints the input if in debug mode, skips otherwise
    :param input_str:
    :return:
    """

    if __debug__:
        print(input_str)


def print_rule(thick=False, line_length=70):
    if thick:
        symbol = '='
    else:
        symbol = '-'
    dprint(symbol*line_length)


def shell_hilite(src_string, color, bold=True):
    color = color.lower()
    colors = {
        'black':    (30, 40),
        'red':      (31, 41),
        'green':    (32, 42),
        'yellow':   (33, 43),
        'blue':     (34, 44),
        'magenta':  (35, 45),
        'cyan':     (36, 46),
        'white':    (37, 47)
    }
    if color not in colors.keys():
        raise UtilsPrintException("Unknown color {}".format(color))
    color_params = [str(colors[color][0])]
    if bold:
        color_params.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(color_params), src_string)


def html_hilite(src_string, color):
    color = color.lower()
    colors = {
        'silver': 	'#C0C0C0',
        'gray': 	'#808080',
        'black': 	'#000000',
        'red': 	    '#FF0000',
        'maroon': 	'#800000',
        'yellow': 	'#FFFF00',
        'olive': 	'#808000',
        'lime': 	'#00FF00',
        'green': 	'#008000',
        'aqua': 	'#00FFFF',
        'teal': 	'#008080',
        'blue': 	'#0000FF',
        'navy': 	'#000080',
        'fuchsia': 	'#FF00FF',
        'purple': 	'#800080'
    }
    if color not in colors.keys():
        raise UtilsPrintException("Unknown color {}".format(color))
    return '<span style="color:{}">{}</span>'.format(colors[color], src_string)