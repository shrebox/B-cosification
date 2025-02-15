configs = {
    (
        f"{sample_size}_{grid_size}x{grid_size}{rescale_suffix}{conf_thresh_suffix}{striding_suffix}{neg_suffix}"
    ): dict(
        n_imgs=grid_size * grid_size,
        sample_size=sample_size,
        do_rescale=do_rescale,
        conf_thresh=conf_thresh,
        striding=striding,
        neg=neg,
    )
    for sample_size in [500, 250, 50]
    for grid_size in [2, 3]
    for rescale_suffix, do_rescale in [("_rescale", True), ("", False)]
    for conf_thresh_suffix, conf_thresh in [("_noconfthresh", 0), ("", 0.5)]
    for striding_suffix, striding in [
        ("", "none"),
        *([(f"-stride={s}", f"{s}") for s in [4, 14, 28, 56, 112]]),
    ]
    for neg_suffix, neg in [("", False), ("-neg", True)]
}
