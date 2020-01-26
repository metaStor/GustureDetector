import config as cfg

dict_cfg = cfg.__dict__

for key in dict_cfg.keys():
    if key[0].isupper():
        print('{}: {}\n'.format(key, dict_cfg[key]))


