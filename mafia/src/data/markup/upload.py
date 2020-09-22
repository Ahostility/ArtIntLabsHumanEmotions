from ...dirs import DIR_DATA_RAW
MARKUP_PATH = DIR_DATA_RAW / 'origin_markup.csv'
WEB_MARKUP_PATH = DIR_DATA_RAW / 'web_origin_markup.csv'

if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    from .columns import *

    TIMILINE_COLS = [START, END]
    URL = 'https://docs.google.com/spreadsheet/ccc?key=1FBzMgIyHwBZjK7LXHRejphz5x802ytXdIyRwWFrUw1A&output=csv'

    def seconds(m: str, s: str) -> float:
        """Parses minutes and seconds to seconds.
        """
        
        return 60 * float(m) + float(s)

    def parse(df: pd.DataFrame) -> pd.DataFrame:
        """Parses the string format of a `minutes:seconds` to the numeric format of a `seconds`.
        """
        tls = [[seconds(*start[-4:].split(':')), seconds(*end[-4:].split(':'))]
                for start, end in df[TIMILINE_COLS].values.astype(np.str)]
        df.loc[df.index.values, TIMILINE_COLS] = tls
        return df

    parse(
        pd.read_csv(URL)
            .dropna()
            .astype(int, errors='ignore')
            .drop(COMMENT, axis=1)
            .set_index(ID)
        ).to_csv(MARKUP_PATH)

    df = pd.read_csv(URL)\
        .dropna()\
        .astype(int, errors='ignore')\
        .set_index(ID)
    df = df[df['COMMENT'].astype(str).str.contains('Int')]\
        .drop(COMMENT, axis=1)
    # df = df.query(f'{RATING_VIDEO} >= {6} and {RATING_AUDIO} >= {0}')
    parse(df).to_csv(WEB_MARKUP_PATH)
