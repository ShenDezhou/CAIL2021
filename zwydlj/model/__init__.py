from model.qa.qa import Model, ModelL, ModelLC, ModelX, ModelXC

model_list = {
    "Model": Model,
    "ModelL": ModelL,
    "ModelLC": ModelLC,
    "ModelX": ModelX,
    "ModelXC": ModelXC
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
