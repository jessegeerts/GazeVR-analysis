import load_data as ld
import os


class Subject:
    def __init__(self, n_subject, pilot=True):
        self.data_dir = './Data/Pilot/'+str(n_subject) if pilot is True else './Data/'+str(n_subject)
        self.ed = ld.load_event_data(os.path.join(self.data_dir, 'events.csv'))
        self.hd = ld.load_raw_head_data(os.path.join(self.data_dir, 'head.csv'))
        self.trajectories_df = ld.construct_df_trajectories_per_trial(self.hd, self.ed)
        self.trials = ld.construct_df_trial_info(self.trajectories_df, self.ed)