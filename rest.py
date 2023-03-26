from flask import Flask, jsonify, request
from joblib import load
# from transformers import  AutoTokenizer, AutoModel
# from torch import Tensor, nn
# from torch import nn
# import torch


# def tensor_masking(tensor: Tensor, mask: Tensor, value: float = 0.0) -> Tensor:
#     return tensor.masked_fill((~(mask.bool())).unsqueeze(-1), value)

# class GlobalMaskedPooling(nn.Module):
#     POOLING_TYPES = ("mean", "max")

#     def __init__(
#         self,
#         pooling_type: str = "mean",
#         dim: int = 1,
#         normalize: bool = False,
#         length_scaling: bool = False,
#         scaling_square_root: bool = False,
#         embedding_masking: bool = True,
#     ):
#         super().__init__()

#         if pooling_type not in self.POOLING_TYPES:
#             raise ValueError(
#                 f"{pooling_type} - is unavailable type." f' Available types: {", ".join(self.POOLING_TYPES)}'
#             )

#         if dim < 0:
#             raise ValueError("Dimension (dim parameter) must be greater than zero")

#         self.pooling_type = pooling_type
#         self.dim = dim

#         self.normalize = normalize
#         self.length_scaling = length_scaling
#         self.scaling_square_root = scaling_square_root

#         self.embedding_masking = embedding_masking

#         if self.pooling_type == "max":
#             self.mask_value = -float("inf")
#         else:
#             self.mask_value = 0.0

#     def forward(self, tensor: Tensor, pad_mask: Tensor) -> Tensor:
#         lengths = pad_mask.sum(self.dim).float()

#         if self.embedding_masking:
#             tensor = tensor_masking(tensor, pad_mask, value=self.mask_value)

#         if self.pooling_type == "mean":
#             scaling = tensor.size(self.dim) / lengths
#         else:
#             scaling = torch.ones(tensor.size(0), device=tensor.device)

#         if self.length_scaling:
#             lengths_factor = lengths
#             if self.scaling_square_root:
#                 lengths_factor = lengths_factor**0.5
#             scaling /= lengths_factor

#         scaling = scaling.masked_fill(lengths == 0, 1.0).unsqueeze(-1)

#         if self.pooling_type == "mean":
#             tensor = tensor.mean(self.dim)
#         else:
#             tensor, _ = tensor.max(self.dim)

#         tensor *= scaling

#         if self.normalize:
#             tensor = F.normalize(tensor)

#         return tensor

#     def extra_repr(self) -> str:

#         description = [
#             f'pooling_type="{self.pooling_type}"',
#             f"normalize={self.normalize}",
#             f"length_scaling={self.length_scaling}",
#             f"scaling_square_root={self.scaling_square_root}",
#         ]

#         description_message = ",\n".join(description)

#         return

# class SentimentClassifier(nn.Module):
#     def __init__(self):
#         super(SentimentClassifier, self).__init__()
#         self.model_dim = 1024
#         self.relu = torch.nn.ReLU()
#         self.extra_linear = torch.nn.Linear(self.model_dim, self.model_dim)
#         self.classification_head = torch.nn.Linear(self.model_dim, 8)
#         self.pooling = GlobalMaskedPooling()

#     def forward(self, model_output, attention_mask):
#         logits = self.pooling(model_output.last_hidden_state, attention_mask)
#         logits = self.relu(self.extra_linear(logits))
#         return self.classification_head(logits)
    
# map_location = 'cpu'
# checkpoint = torch.load('pretrained_models/model_ruBert_large.bin', map_location=map_location)
# sentiment_classifier_model = SentimentClassifier()
# sentiment_classifier_model.load_state_dict(checkpoint)

# classes_mapping = {7: 'women', 0: 'born_place', 3: 'man', 6: 'other', 2: 'lgbt', 4: 'migrant', 1: 'child', 5: 'no_hate'}

# rubert_large_model = AutoModel.from_pretrained('pretrained_models/ruBert-large')
# rubert_large_tokenizer = AutoTokenizer.from_pretrained('pretrained_models/ruBert-large')


# def bert_predict(model_name, text):
#     if model_name == 'rubert_large':
#         model, tokenizer = rubert_large_model, rubert_large_tokenizer

#     embeds = tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(input_ids=embeds['input_ids'], attention_mask=embeds['attention_mask'])
#     pred = sentiment_classifier_model(model_output, embeds['attention_mask'])
#     raw_pred = pred.argmax(1).detach().numpy()[0]
#     target_group = classes_mapping[raw_pred]
    
#     return target_group


trained = load("pretrained_models/pipe.joblib")

app = Flask(__name__)

def logistic_regressor_predict(text):
    prediction = trained.predict([text])
    return prediction[0]

@app.route("/predict")
def ruhabe_index():
    text = request.args.get("text", "DEFAULT")
    model = request.args.get("selected_model", "DEFAULT")

    # if model == 'BERT':
        # return jsonify({'group': bert_predict('rubert_large', text)})
    return jsonify({'group': logistic_regressor_predict(text)})

@app.route("/hello")
def hello():
    return "<p>hello!</p>"

if __name__ == '__main__':
    print('Hello from rest api!')
    app.run(debug=True, host="0.0.0.0", port=8000)
