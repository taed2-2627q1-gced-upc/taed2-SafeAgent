from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("models/encoders/codebert__command__2__a740d875")
print(model.config.id2label)