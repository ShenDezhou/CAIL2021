from model.qa.qa import Model, ModelL, ModelS, ModelX, ModelXC

model_list = {
    "Model": Model,
    "ModelL": ModelL,
    "ModelS": ModelS,
    "ModelX": ModelX,
    "ModelXC": ModelXC
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
