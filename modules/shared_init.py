from modules import shared


def initialize():
    """
    Initializes fields inside the shared module in a controlled manner.
    Should be called early because some other modules you can import mingt need these fields to be already set.
    """

    from modules import options, shared_options

    shared.options_templates = shared_options.options_templates
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts
    try:
        shared.opts.load(shared.config_filename)
    except FileNotFoundError:
        pass

    from modules_forge.mmgp_profiles import get_mmgp_quantization, get_mmgp_residency_components

    # Populate merged MMGP settings from older config keys before the settings
    # UI is built, so applying an unrelated setting cannot reset them.
    if "forge_memory_quantization" not in shared.opts.data:
        quantize, quantization_type = get_mmgp_quantization(shared.opts)
        shared.opts.data["forge_memory_quantization"] = quantization_type if quantize else "disabled"
    if "forge_memory_residency_components" not in shared.opts.data:
        shared.opts.data["forge_memory_residency_components"] = get_mmgp_residency_components(shared.opts)

    from modules import devices
    shared.device = devices.device

    from modules import shared_state
    shared.state = shared_state.State()

    from modules import styles
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    from modules import memmon, devices
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()
