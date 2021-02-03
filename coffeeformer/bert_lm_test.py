from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./model/bbpe",
    tokenizer="./model/bbpe"
)

mask_pred = fill_mask("在一件申请需要分案的情况下，对分案的审查包括对分案申请的审<mask>")
print(mask_pred)

mask_pred = fill_mask("被侵害人，是因自己的人身、财产、名<mask>")
print(mask_pred)