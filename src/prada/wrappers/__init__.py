C2I_MODELS = ["var_d20", "var_d30", "rar_l", "rar_xxl", "hmar_d20", "hmar_d30", "llamagen_b256", "llamagen_l256"]
T2I_MODELS = ["infinity_2b", "janus_1b", "llamagen_xlstage2", "switti_1024"]


def get_wrapper(model: str, **kwargs):
    """Load wrapper, with suffix indicating variant."""
    print(f"Loading wrapper for model {model} with kwargs {kwargs}...")
    if "_" in model:
        model, variant = model.split("_")
    else:
        variant = None

    if model == "var":
        from .var import VARWrapper

        return VARWrapper(model_depth=int(variant.removeprefix("d")), **kwargs)

    elif model == "llamagen":
        from .llamagen import LlamaGenWrapper

        if variant == "xl":
            return LlamaGenWrapper(
                gpt_model_name=variant, gpt_type="t2i", gpt_stage_t2i="stage2", image_size=512, **kwargs
            )
        else:
            return LlamaGenWrapper(gpt_model_name=variant.removesuffix("256"), **kwargs)

    elif model == "rar":
        from .rar import RARWrapper

        return RARWrapper(rar_model_size=variant, **kwargs)

    elif model == "hmar":
        from .hmar import HMARWrapper

        return HMARWrapper(model_depth=variant, **kwargs)

    elif model == "infinity_2b":
        from .infinity import InfinityWrapper

        return InfinityWrapper(**kwargs)

    elif model == "janus":
        from .janus import JanusWrapper

        return JanusWrapper(variant=variant, **kwargs)

    elif model == "switti":
        from .switti import SwittiWrapper

        return SwittiWrapper(variant, **kwargs)

    else:
        raise NotImplementedError(f"Model {model} is not implemented.")
