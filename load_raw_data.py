import pandas as pd


def parse_dates(timestamp):
    """
    parse_dates(timestampstring) takes a timestamp string formatted as Year-month-dayThour:min:sec.decimals+Timezone
    and converts it into a datetime.datetime format, ignoring the timezone and the last decimal, keeping a microsecond
    precision.
    """
    return pd.datetime.strptime(timestamp[:26], '%Y-%m-%dT%H:%M:%S.%f')


def load_event_data(filename):
    """
    load_event_data(filename)

    :param filename:
    :return: pandas time-series
    """
    ts = pd.read_csv(filename,
                     header=None,
                     parse_dates=[0],
                     names=['Timestamp', 'Name', 'Value1', 'Value2'],
                     date_parser=parse_dates,
                     delimiter=' ')
    ts = ts.set_index(['Timestamp'])
    return ts


def load_head_data(filename):
    """
    load_head_data(filename) reads a csv file containing the head position time series data, and casts it into a
    pandas time-series

    :param filename:
    :return: pandas time-series containing a 4x4 matrix representing the rotation and translation
    """
    hd = pd.read_csv(filename)
    hd['Timestamp'] = hd['Timestamp'].apply(parse_dates)
    hd = hd.set_index('Timestamp')
    return hd


def get_location_dataframe(matrix_dataframe):
    """
    Extracts the X, Y and Z location from the 4x4 matrix dataframe above and places them
    in a smaller dataframe with just the locations per timestamp.

    :param pd.DataFrame matrix_dataframe:
    :return: pd.DataFrame
    """

    df_location = matrix_dataframe.loc[:, ['Value.M41', 'Value.M42', 'Value.M43']]
    df_location.columns = ['X', 'Y', 'Z']
    return df_location
