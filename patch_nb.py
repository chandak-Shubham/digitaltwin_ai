import json
import os

with open('model_comparison_template.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if "y = groups[TARGET].max().astype('float32').to_numpy()" in line:
                source[i] = "    y = np.stack([group[TARGET].astype('float32').to_numpy() for _, group in groups])\n"
            if "return self.head(out[:, -1]).squeeze(1)" in line:
                source[i] = "        return self.head(out).squeeze(2)\n"
            if "def scores(y, p):" in line:
                source[i] = "def scores(y, p):\n    y = y.flatten() if y.ndim > 1 else y\n    p = p.flatten() if p.ndim > 1 else p\n"

        if any("def benchmark(name, df):" in line for line in source):
            new_source = [
                "def benchmark(name, df):\n",
                "    train_df, test_df = split_vehicles(df)\n",
                "    train_matrix, test_matrix = preprocess(train_df, test_df)\n",
                "    X, y = sequences(train_df, train_matrix)\n",
                "    Xt, yt = sequences(test_df, test_matrix)\n",
                "    y_max = y.max(axis=1)\n",
                "    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)\n",
                "    output = []\n",
                "    for model_name in MODEL_NAMES:\n",
                "        best_auc, best = -np.inf, None\n",
                "        for params in ParameterGrid(GRIDS[model_name]):\n",
                "            fold_auc = []\n",
                "            for a, b in cv.split(X, y_max):\n",
                "                if model_name == 'lightgbm':\n",
                "                    model = lgb.LGBMClassifier(objective='binary', class_weight='balanced', verbosity=-1, random_state=SEED, **params)\n",
                "                    model.fit(X[a].reshape(len(a), -1), y_max[a])\n",
                "                    p = model.predict_proba(X[b].reshape(len(b), -1))[:, 1]\n",
                "                    fold_auc.append(roc_auc_score(y_max[b], p))\n",
                "                elif model_name == 'lstm':\n",
                "                    p = fit_lstm(X[a], y[a], X[b], params)\n",
                "                    fold_auc.append(roc_auc_score(y[b].flatten(), p.flatten()))\n",
                "            if np.mean(fold_auc) > best_auc:\n",
                "                best_auc, best = float(np.mean(fold_auc)), params\n",
                "        if model_name == 'lightgbm':\n",
                "            final = lgb.LGBMClassifier(objective='binary', class_weight='balanced', verbosity=-1, random_state=SEED, **best)\n",
                "            final.fit(X.reshape(len(X), -1), y_max)\n",
                "            p = final.predict_proba(Xt.reshape(len(Xt), -1))[:, 1]\n",
                "            yt_score = yt.max(axis=1)\n",
                "        elif model_name == 'lstm':\n",
                "            p = fit_lstm(X, y, Xt, best)\n",
                "            yt_score = yt\n",
                "        result = {'dataset': name, 'model': model_name, 'cv_auc': best_auc, 'best_params': best, **scores(yt_score, p)}\n",
                "        output.append(result)\n",
                "        print(name, model_name, 'CV AUC=', round(best_auc, 4), 'TEST AUC=', round(result['roc_auc'], 4), 'F1=', round(result['f1'], 4), 'Best:', best)\n",
                "    return output\n",
                "\n",
                "results = []\n",
                "for name, path in DATASETS.items():\n",
                "    results.extend(benchmark(name, load_dataset(path)))\n",
                "results_df = pd.DataFrame(results)\n"
            ]
            cell['source'] = new_source

with open('model_comparison_template.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched successfully")
