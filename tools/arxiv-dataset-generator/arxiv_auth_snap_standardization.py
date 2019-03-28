# transforms author name string into standardized form according to process
#  used in Stanford SNAP package, i.e.:
#  https://github.com/snap-stanford/snap/blob/master/snap-core/util.cpp
#   in method  'TStrUtil::GetStdName'


def to_snap_standardized_auth_name(str):
    str = str.lower().replace('\n', ' ').replace('.', ' ')
    str = str[:findDigit(str)].strip()
    str = str[:findChar(str, '(')].strip()
    if (findChar(str, ')') != len(str)):
        return None
    if str.find('figures') != -1:
        return None
    if str.find('macros') != -1:
        return None
    if str.find('univ') != -1:
        return None
    if str.find('institute') != -1:
        return None
    str = filterAlpha(str)
    tokens = str.split()
    if len(tokens) > 0 and tokens[-1] == 'jr':
        tokens = tokens[:-1]
    if len(tokens) < 2:
        return None

    lastName = tokens[-1]
    if not lastName[0].isalpha or len(lastName) == 1:
        return None
    if not tokens[0][0].isalpha():
        return None
    return '{}_{}'.format(lastName, tokens[0][0])


def findDigit(str):
    ix = 0
    for c in str:
        if c.isdigit() or c == '#':
            return ix
        ix += 1
    return ix


def findChar(str, ch):
    ix = 0
    for c in str:
        if c == ch:
            return ix
        ix += 1
    return ix


def filterAlpha(str):
    newStr = ''
    for c in str:
        if c.isalpha() or c.isspace() or c == '-':
            newStr += c
    return newStr.strip()


def tests():
    oklist = ['B. Anton', 'Bulb Anton', 'Bulb \nAnton', 'Bulb Anton 3 dupa',
              'Bulb Anton # dupa', 'Bulb Anton ( dupa', 'Bulb Anton (dupa)',
              'Bulb /Anton', 'Bulb! ?Anton^^', ' & Bulb *Anton ..,..',
              'Bulb Anton jr.', 'Bulb Anton @ JR', 'Bulb - Anton',
              'Bulb Jaako Anton', 'Bulb  Jose Maia de la Anton']
    nolist = ['Bulb Anton )', 'Bulb Figures', 'Bulb Macros', 'Bulb Univ', '',
              'Bulb University', 'Bulb Institute', 'Bulb ', 'Bulb jr.', '  ',
              'Bulb * *  *  *', ' - Bulb', 'Bulb - ', '1234', 'jr']

    for tstr in oklist:
        pstr = to_snap_standardized_auth_name(tstr)
        if pstr != 'anton_b':
            print('FAILED:\n\tINPUT:   {}\n\tOUTPUT:  {}\n\tEXPECTED: {}\n'
                  .format(tstr, pstr, 'anton_b'))
    for tstr in nolist:
        pstr = to_snap_standardized_auth_name(tstr)
        if pstr is not None:
            print('FAILED:\n\tINPUT:   {}\n\tOUTPUT:  {}\n\tEXPECTED: None\n'
                  .format(tstr, pstr))


if __name__ == '__main__':
    tests()
