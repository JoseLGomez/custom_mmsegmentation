pseudolabeling = dict(mode='mpt',
                      accumulation='update_best_score',
                      collaboration='cotraining',
                      sorting='confidence',
                      number=200,
                      init_tgt_port=0.5,
                      max_tgt_port=0.7,
                      tgt_port_step=0.05,
                      init_tgt_port_B=0.5,
                      max_tgt_port_B=0.7,
                      tgt_port_step_B=0.05,
                      min_conf=0.5,
                      max_conf=0.9
                        )       