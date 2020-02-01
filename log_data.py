import pandas as pd

cols = ["num_conn","startTimet","orig_pt","resp_pt1","orig_ht","resp_ht",
"duration","protocol_type","resp_pt2","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urg","hot","num_failed_logins",
"logged_in","num_compromised","root_shell","su_attempted",
"num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_hot_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate_sec","rerror_rate","srv_error_rate_sec","same_srv_rate","diff_srv_rate",
"dst_host_diff_srv_rate","count_100","srv_count_100","same_srv_rate_100","diff_srv_rate_100",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rat","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate"]


def get_server_logs_df():
  return pd.read_csv('trafAld.list', delimiter=' ', names=cols, header=None)

def save_file():
  get_server_logs_df().to_csv('realtime_test_dataset.csv', index=False)

if __name__ == '__main__':
  save_file()
