# batch_ratio: tuple with source/pseudolabel proportion on batch
uda = dict(mix_batch=True, type='MixBatch', batch_ratio=(1,3),
           classmix=dict(apply=False, apply_rcs=False, apply_rcs_chance=0.5, blur=True, freq=0.5,
                         color_jitter_strength=0.2,
                         color_jitter_probability=0.2))