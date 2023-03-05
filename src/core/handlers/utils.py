from re import sub


class SubRegexBuilder(str):
    """
    It extends the str class to implement a builder pattern 
    that allows the sub function to be applied multiple 
    times.
    """

    def __new__(cls, *args, **kwargs):
        newobj = str.__new__(cls, *args, **kwargs)
        newobj.sub = lambda fro, to: SubRegexBuilder(sub(fro, to, newobj))
        return newobj