HMM_VARIANTS = {
    'GHMM': ('Gaussian HMM', 'gHMM', 0, False, False),
    # 'DiagGHMM': ('Gaussian HMM\n(diagonal covariance)', '', 0, False, True),
    'IDGHMM': ('Gaussian HMM\n(input transitions)', 'gHMM(I)', 0, True, False),
    'LRHMM': ('Linear-Gaussian HMM', 'lgHMM', 1, False, False),
    'IDLRHMM': ('Linear-Gaussian HMM\n(input transitions)', 'lgHMM(I)', 1, True, False),
    'IDARHMM': ('AR linear-Gaussian HMM\n(input transitions)', 'ARlgHMM(I)', 2, True, False),
}
