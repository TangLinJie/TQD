import os


root = './'
out_root = '../results_new'
files = os.listdir(root)
for file in files:
    with open(os.path.join(root, file), 'rb') as f:
        lines = [line.strip() for line in f.readlines()]
    with open(os.path.join(out_root, file.replace('ts_', '')), "wb") as f:
        f.write(b'\r\n'.join(lines))
    # os.rename(os.path.join(root, file), os.path.join(root, file.replace('res_ts_ts', 'res')))