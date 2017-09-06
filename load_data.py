import pandas as pd
import numpy as np
from transforms3d.euler import mat2euler


def construct_df_trajectories_per_trial(hd, ed):
    """
    This function takes in the Head Data (hd) and Event Data (ed) frames, and constructs a DataFrame that is indexed
    on time stamp, but contains just the trajectories between the appearance of the target and the end of each trial.

    :param hd: Pandas DataFrame containing the head data
    :param ed: Pandas DataFrame containing the event data
    :return:
    """

    target_times = ed[(ed['Name'] == 'TargetLeft') | (ed['Name'] == 'TargetRight')].index

    end_trial_indices = [ed.index.get_loc(trial) + 1 for trial in target_times]
    end_trial_times = ed.iloc[end_trial_indices].index  # the corresponding timestamps

    target_sides = ed[ed.Name.str.get(0).isin(['T'])].reset_index()

    trajectories = []
    for i, (start, end) in enumerate(zip(target_times, end_trial_times)):
        trial_trajectory = hd.loc[start:end]
        trial_trajectory = trial_trajectory.resample('0.01S').pad()
        trial_trajectory.loc[:, 'Trial number'] = i
        trial_trajectory.loc[:, 'Target side'] = target_sides.iloc[i]['Name']
        trial_trajectory['Trial time'] = trial_trajectory.index - trial_trajectory.index[0]
        trajectories.append(trial_trajectory)

    trajectories_df = pd.concat(trajectories).sort_index()

    # convert to matrices and then to angles
    list_of_matrices = [series2mat4(trajectories_df.iloc[x]) for x in range(trajectories_df.shape[0])]
    angles = np.array([np.degrees(mat2euler(mat, 'syzx')) for mat in list_of_matrices])  # retrieve euler angles
    angles_df = pd.DataFrame(angles, index=trajectories_df.index, columns=['Y rotation', 'Z rotation', 'X rotation'])
    trajectories_df = trajectories_df.join(angles_df)

    trial_starts = trajectories_df[trajectories_df['Trial time'] == trajectories_df.iloc[1]['Trial time']]
    zero_y = trial_starts['Y rotation'].mean()
    zero_z = trial_starts['Z rotation'].mean()
    trajectories_df['Centred Y angle'] = trajectories_df['Y rotation'] - zero_y
    trajectories_df['Centred Z angle'] = trajectories_df['Z rotation'] - zero_z
    return trajectories_df


def construct_df_trial_info(trajectories_df, ed):

    # Determine reaction times
    rt = []
    for i in trajectories_df['Trial number'].unique():
        idx = trajectories_df['Trial number'] == i
        rt.append(trajectories_df[idx]['Trial time'].max())

    trials = pd.DataFrame(index=trajectories_df['Trial number'].unique(),
                          columns=['rt'],
                          data=np.array(np.array(rt)))
    trials.index.name = 'Trial'
    trials['Target side'] = ed[ed.Name.str.get(0).isin(['T'])].reset_index()['Name']
    trials['Reaction time (ms)'] = trials['rt'].apply(lambda x: x.microseconds / 1000)

    # determine trajectory endpoints
    startpoints_y = []
    endpoints_y = []
    startpoints_z = []
    endpoints_z = []
    for i in trajectories_df['Trial number'].unique():
        idx = trajectories_df['Trial number'] == i
        startpoints_y.append(trajectories_df[idx].iloc[1]['Centred Y angle'])
        endpoints_y.append(trajectories_df[idx].iloc[-1]['Centred Y angle'])
        startpoints_z.append(trajectories_df[idx].iloc[1]['Centred Z angle'])
        endpoints_z.append(trajectories_df[idx].iloc[-1]['Centred Z angle'])

    trials['Starting points Y'] = startpoints_y
    trials['Movement endpoints Y'] = endpoints_y
    trials['Starting points Z'] = startpoints_z
    trials['Movement endpoints Z'] = endpoints_z

    trial_results = ed[(ed['Name'] == 'Neutral') |
                       (ed['Name'] == 'Missed') |
                       (ed['Name'] == 'Hit') |
                       (ed['Name'] == 'Penalty')]
    trials['Outcome'] = np.array(trial_results['Name'])
    return trials


def load_event_data(filename):
    """
    load_event_data(filename)

    :param filename:
    :return: pandas time-series
    """
    ed = pd.read_csv(filename,
                     header=None,
                     parse_dates=[0],
                     names=['Timestamp', 'Name', 'Value1', 'Value2'],
                     date_parser=_parse_dates,
                     delimiter=' ')
    ed = ed.set_index(['Timestamp'])
    return ed


def load_raw_head_data(filename):
    """
    load_head_data(filename) reads a csv file containing the head position time series data, and casts it into a
    pandas time-series

    :param filename:
    :return: pandas time-series containing a 4x4 matrix representing the rotation and translation
    """
    hd = pd.read_csv(filename)
    hd['Timestamp'] = hd['Timestamp'].apply(_parse_dates)
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


def _parse_dates(timestamp):
    """
    parse_dates(timestampstring) takes a timestamp string formatted as Year-month-dayThour:min:sec.decimals+Timezone
    and converts it into a datetime.datetime format, ignoring the timezone and the last decimal, keeping a microsecond
    precision.
    """
    return pd.datetime.strptime(timestamp[:26], '%Y-%m-%dT%H:%M:%S.%f')


def series2mat4(hp):
    """
    Reads the head position as coded by Bonsai from a DataFrame and converts it to a 2D NumPy array

    :param hp: Pandas Series containing an affine matrix describing position and orientation
    :return:
    """
    return np.array([[hp.loc['Value.M11'], hp.loc['Value.M21'], hp.loc['Value.M31'], hp.loc['Value.M41']],
                     [hp.loc['Value.M12'], hp.loc['Value.M22'], hp.loc['Value.M32'], hp.loc['Value.M42']],
                     [hp.loc['Value.M13'], hp.loc['Value.M23'], hp.loc['Value.M33'], hp.loc['Value.M43']],
                     [hp.loc['Value.M14'], hp.loc['Value.M24'], hp.loc['Value.M34'], hp.loc['Value.M44']]])

