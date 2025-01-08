import clip

def get_model(model_name, *args, **kwargs):
    if model_name == "clip":
        return clip.load(*args, **kwargs)