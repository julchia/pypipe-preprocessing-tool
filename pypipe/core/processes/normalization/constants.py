import re


WHITE_SPACE_REGEX = re.compile(r'\s+')

PUNCTUTATION_REGEX = re.compile(r'[^\w\s]')

DIGIT_REGEX = re.compile(r'\d+')

SPECIAL_CHARS_REGEX = re.compile(r'[a-zA-ZáéíóúÁÉÍÓÚñÑüÜàèìòùÀÈÌÒÙäëïöüÄËÏÖÜâêîôûÂÊÎÔÛ]+')

MENTION_REGEX = re.compile(r'(@|#)[A-Za-z0-9]+')

# source: https://gist.github.com/dperini/729294
URL_REGEX = re.compile(
    r"(?:^|(?<![\w\/\.]))"
    # protocol identifier
    # r"(?:(?:https?|ftp)://)"  <-- alt?
    r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?" r"(?:"
    # IP address exclusion
    # private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name
    r"(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))" r"|" r"(?:(localhost))" r")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:\/[^\)\]\}\s]*)?",
    # r"(?:$|(?![\w?!+&\/\)]))",
    # @jfilter: I removed the line above from the regex because I don't understand what it is used for, maybe it was useful?
    # But I made sure that it does not include ), ] and } in the URL.
    flags=re.UNICODE | re.IGNORECASE,
)

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

SINGLE_WORD_REGEX = re.compile(r'(?<!\S)[^aeiouy](?!\S)')

DUPLICATED_LETTER_REGEX = re.compile(r'(?!l|r)(.)\1{1,}')

ISOLATED_CONSONANT_REGEX = re.compile(r'(?<=\s)[bcdfghjklmnpqrstvwxyz]{2,}(?=\s)')

RE_REGEX = re.compile(r'(?<!\S)re(?!\S)')

Q_REGEX = {
    "que": re.compile(r'\s?(ke|k|qe|q)\s'),
    "quie": re.compile(r'\s?(kie|qie)')
}

LAUGHT_REGEX = {
    "ja": re.compile(r'((ja|aj|ha){3,})'),
    "je": re.compile(r'((je|ej|he){3,})'),
    "ji": re.compile(r'((ji|ij){3,})'),
    "jo": re.compile(r'((jo|oj){3,})'),
    "ju": re.compile(r'((ju|uj){3,})')
}
