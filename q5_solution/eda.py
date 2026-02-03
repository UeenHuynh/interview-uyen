#!/usr/bin/env python3
import os
from pathlib import Path
from collections import Counter
import pandas as pd

base = Path('/home/neeyuhuynh/Desktop/me/genesolution/interview_data/Supplement_datasets')
files = ['FLEN.csv', 'EM.csv', 'NUCLEOSOME.csv']

a = {}
for f in files:
    df = pd.read_csv(base / f)
    df = df.set_index(df.columns[0])
    a[f] = df.T

samples = list(a['FLEN.csv'].index)
labels = [s.split('_')[0] for s in samples]
label_counts = Counter(labels)

# Build combined feature table (concat columns)
combined = pd.concat([a['FLEN.csv'], a['EM.csv'], a['NUCLEOSOME.csv']], axis=1)
combined.columns = combined.columns.astype(str)

out_dir = Path('/home/neeyuhuynh/Desktop/me/genesolution/q5_solution/out')
out_dir.mkdir(parents=True, exist_ok=True)

# Save basic stats
stats_path = out_dir / 'basic_stats.txt'
with open(stats_path, 'w', encoding='utf-8') as f:
    f.write('Samples\t%s\n' % len(samples))
    f.write('Classes\t%s\n' % ','.join(sorted(label_counts)))
    f.write('Class_counts\t%s\n' % dict(label_counts))
    for name, df in a.items():
        vars_ = df.var(axis=0)
        f.write(f"{name}_shape\t{df.shape}\n")
        f.write(f"{name}_var_min\t{vars_.min():.6g}\n")
        f.write(f"{name}_var_median\t{vars_.median():.6g}\n")
        f.write(f"{name}_var_max\t{vars_.max():.6g}\n")
    f.write(f"Combined_shape\t{combined.shape}\n")

# Plot class counts
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    x = list(label_counts.keys())
    y = [label_counts[k] for k in x]
    plt.bar(x, y, color='#4C78A8')
    plt.title('Class counts')
    plt.ylabel('Samples')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / 'class_counts.png', dpi=150)
    plt.close()
except Exception:
    pass

# PCA plots (if sklearn available)
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    def pca_plot(X, y, title, out_path):
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,5))
        for cls in sorted(set(y)):
            idx = [i for i, v in enumerate(y) if v == cls]
            plt.scatter(Xp[idx,0], Xp[idx,1], s=10, alpha=0.7, label=cls)
        plt.title(title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    pca_plot(a['FLEN.csv'], labels, 'PCA - FLEN', out_dir / 'pca_flen.png')
    pca_plot(a['EM.csv'], labels, 'PCA - EM', out_dir / 'pca_em.png')
    pca_plot(a['NUCLEOSOME.csv'], labels, 'PCA - NUCLEOSOME', out_dir / 'pca_nucleosome.png')
    pca_plot(combined, labels, 'PCA - Combined', out_dir / 'pca_combined.png')
except Exception:
    pass
