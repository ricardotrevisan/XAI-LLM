import ollama
import shap
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 1. Prompts
prompts = [
    "Explique a fotossíntese em termos simples.",
    "O que é inteligência artificial?",
    "Como funciona o blockchain?"
]

# 2. Função para chamar LLM
def query_llm(prompt_text):
    ollama_model = "deepseek-r1"
    response = ollama.generate(model=ollama_model, prompt=prompt_text)
    return response['response']

# 3. Obter respostas do LLM
responses = [query_llm(p) for p in prompts]

# 4. Inicializar modelo de embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 5. Pré-processamento: embeddings completos
X_words = [p.split() for p in prompts]
Y_words = [r.split() for r in responses]

X_emb_words = [embed_model.encode(words) for words in X_words]
Y_emb_words = [embed_model.encode(words) for words in Y_words]

max_words = max(len(words) for words in X_words)
feature_dim = X_emb_words[0].shape[1]

X_emb_full = []
for emb in X_emb_words:
    num_words = emb.shape[0]
    padded = np.zeros((max_words, feature_dim))
    padded[:num_words, :] = emb
    X_emb_full.append(padded.flatten())
X_emb_full = np.array(X_emb_full)

Y_emb_mean = np.array([emb.mean(axis=0) for emb in Y_emb_words])

# 6. Wrapper determinístico para SHAP
def model_wrapper_word(X_input):
    outputs = []
    for x_vec in X_input:
        x_words_emb = x_vec.reshape(max_words, feature_dim)
        actual_words_emb = x_words_emb[np.any(x_words_emb != 0, axis=1)]
        sims = [cosine_similarity(actual_words_emb.mean(axis=0).reshape(1, -1),
                                  Y_emb_mean[j].reshape(1, -1))[0,0]
                for j in range(len(Y_emb_mean))]
        outputs.append(max(sims))
    return np.array(outputs)

# 7. Explicabilidade com SHAP
explainer = shap.KernelExplainer(model_wrapper_word, X_emb_full)
shap_values_full = explainer.shap_values(X_emb_full)

# 8. Agregar SHAP por palavra
shap_values_per_word = []
for i, emb in enumerate(X_emb_words):
    num_words = emb.shape[0]
    shap_vals = shap_values_full[i].reshape(max_words, feature_dim)
    shap_values_per_word.append(shap_vals[:num_words].sum(axis=1))

# 9. Summary plot (concatenado)
shap_values_concat = np.concatenate(shap_values_per_word)
feature_names_concat = []
for i, wlist in enumerate(X_words):
    feature_names_concat.extend([f"{w} (P{i+1})" for w in wlist])
shap_values_concat_2d = shap_values_concat.reshape(1, -1)

shap.summary_plot(shap_values_concat_2d, feature_names=feature_names_concat, plot_type="bar")

# 10. Heatmaps em painel único
num_prompts = len(X_words)
fig, axes = plt.subplots(num_prompts, 1, figsize=(max(6, max(len(p) for p in X_words)), 2*num_prompts))

if num_prompts == 1:
    axes = [axes]

# encontrar valor máximo e mínimo entre todos os prompts para cores consistentes
all_vals = np.concatenate(shap_values_per_word)
vmin, vmax = all_vals.min(), all_vals.max()

for i, prompt_words in enumerate(X_words):
    shap_vals_prompt = shap_values_per_word[i].reshape(1, -1)
    
    sns.heatmap(shap_vals_prompt,
                annot=np.round(shap_vals_prompt, 2),
                fmt=".2f",
                cmap='coolwarm',
                cbar=True if i == 0 else False,
                vmin=vmin, vmax=vmax,  # cores consistentes
                xticklabels=prompt_words,
                yticklabels=[],
                ax=axes[i])
    axes[i].set_title(f"Impacto palavra a palavra: Prompt {i+1}")

plt.tight_layout()
plt.show()
