pseudolabeling = dict(mode='mpt',
                      accumulation='update_best_score',
                      sorting='confidence',
                      number=200,
                      init_tgt_port=0.5,
                      max_tgt_port=0.75,
                      tgt_port_step=0.05,
                      min_conf=0.5,
                      max_conf=0.95)
