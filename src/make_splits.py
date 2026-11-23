"""splits/splits.csv üretir: herkesin kullandığı tek bölme dosyası.

Sütunlar:
  spk_fold : konuşmacıdan bağımsız kat (RAVDESS 0-5, EmoDB 0-4)
  rnd_fold : yalnız RAVDESS için, konuşmacıyı YOK SAYAN rastgele kat (0-5),
             duyguya göre tabakalı. E3 (veri sızıntısı) deneyi bunu kullanır.

Her kat k için: test = k, doğrulama = (k+1) mod K, eğitim = geri kalanlar.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import DATA, EMODB_FOLDS, RAVDESS_FOLDS, SPLITS


def main(seed: int = 42):
    df = pd.read_csv(DATA / "metadata.csv")
    r_map = {f"R{a:02d}": k for k, actors in enumerate(RAVDESS_FOLDS) for a in actors}
    e_map = {f"E{s}": k for k, spk in enumerate(EMODB_FOLDS) for s in spk}
    df["spk_fold"] = [r_map[s] if c == "ravdess" else e_map[s]
                      for s, c in zip(df.speaker, df.corpus)]

    df["rnd_fold"] = -1
    rav = df[df.corpus == "ravdess"]
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(rav, rav.emotion)):
        df.loc[rav.index[te], "rnd_fold"] = k

    SPLITS.mkdir(parents=True, exist_ok=True)
    cols = ["utt_id", "corpus", "speaker", "gender", "emotion", "emotion6", "spk_fold", "rnd_fold"]
    df[cols].to_csv(SPLITS / "splits.csv", index=False)

    # Kontrol: her konuşmacı tek bir spk_fold'da; her kat cinsiyet dengeli
    assert df.groupby("speaker").spk_fold.nunique().max() == 1
    print(df.groupby(["corpus", "spk_fold"]).agg(n=("utt_id", "size"),
                                                  speakers=("speaker", lambda s: " ".join(sorted(set(s)))),
                                                  female=("gender", lambda g: (g == "f").mean().round(2))))
    # Rastgele bölmede aynı oyuncunun kaç kata dağıldığı (sızıntının kanıtı)
    print("rnd_fold: oyuncu başına ortalama kat sayısı =",
          round(df[df.corpus == "ravdess"].groupby("speaker").rnd_fold.nunique().mean(), 2))
    print("Yazıldı:", SPLITS / "splits.csv")


if __name__ == "__main__":
    main()
