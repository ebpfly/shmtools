# %%
import LADPackage
import examples
import shmtools

# %%
# Import condition-based monitoring experimental data.
dataset, damage_states, state_list, fs = examples.data.import_cbm_data_shm()

# %%
# Feature Extraction: Estimate power spectral density via Welch's method.
psd_matrix, f, use_one_sided = shmtools.psd_welch_shm(
    X=dataset,  # <class 'numpy.ndarray'> (required)
    n_win=None,  # typing.Optional[int] (optional)
    n_ovlap=None,  # typing.Optional[int] (optional)
    n_fft=None,  # typing.Optional[int] (optional)
    fs=fs,  # typing.Optional[float] Hz (optional)
    use_one_sided=True  # typing.Optional[bool] (optional)
)

# %%
# Plot power spectral density with various visualization options.
axes = shmtools.plot_psd_shm(
    psd_matrix=psd_matrix,  # <class 'numpy.ndarray'> (required)
    channel=3,  # <class 'int'> (optional)
    is_one_sided=use_one_sided,  # <class 'bool'> (optional)
    f=f,  # typing.Optional[numpy.ndarray] (optional)
    use_colormap=False,  # <class 'bool'> (optional)
    use_subplots=False,  # <class 'bool'> (optional)
    ax=None,  # typing.Optional[matplotlib.axes._axes.Axes] (optional)
    kwargs=None  # Any (required)
)

# %%



