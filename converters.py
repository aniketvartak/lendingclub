
# Converts percentages to floats
def p2f(x):
    if(x.endswith("%")):
        return float(x.strip("%"))/100.
    else:
        print x

# Converts year strings in the "employment years" fields to floats
def ytof(x):
    if(x.endswith('10+ years')):
        return 10.
    elif(x.endswith('< 1 year')):
            return 0.
    elif(x.endswith('1 year')):
        return 1.
    elif(x.endswith('years')):
        return float(x.strip(" years"))
    elif(x.endswith('n/a')):
        return 555.

# Return number of words in a string
def str2words(s):
    return float(len(s.split()))

# Returns a values from "terms" field
def term2f(s):
    if(s.endswith(" months")):
        return float(s.strip(" months"))
