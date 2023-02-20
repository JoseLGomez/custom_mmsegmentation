uda = dict(mix_batch=True, type='MixBatch', batch_ratio=(1,3),
           classmix=dict(apply=True, blur=True, freq=0.5,
                         color_jitter_strength=0.2,
                         color_jitter_probability=0.2),
           imnet_feature_dist = dict(
                                     imnet_feature_dist_lambda=0.005,
                                     imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
                                     imnet_feature_dist_scale_method='ratio',
                                     imnet_feature_dist_scale_min_ratio=0.75,
                                     warmup_step=True,
                                     warmup_iters=30000
                                    )
           )