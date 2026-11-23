| Model | Alt grup | n | UAR (%) |
|---|---|---|---|
| baseline_bilstm | tümü | 1440 | 43.5 |
| baseline_bilstm | normal şiddet | 768 | 38.8 |
| baseline_bilstm | güçlü şiddet | 672 | 50.6 |
| baseline_bilstm | kadın | 720 | 51.4 |
| baseline_bilstm | erkek | 720 | 35.5 |
| wavlm_large_frozen_wls | tümü | 1440 | 79.2 |
| wavlm_large_frozen_wls | normal şiddet | 768 | 75.5 |
| wavlm_large_frozen_wls | güçlü şiddet | 672 | 83.2 |
| wavlm_large_frozen_wls | kadın | 720 | 86.1 |
| wavlm_large_frozen_wls | erkek | 720 | 72.4 |
| wavlm_large_ft2 | tümü | 1440 | 71.5 |
| wavlm_large_ft2 | normal şiddet | 768 | 66.1 |
| wavlm_large_ft2 | güçlü şiddet | 672 | 76.6 |
| wavlm_large_ft2 | kadın | 720 | 77.3 |
| wavlm_large_ft2 | erkek | 720 | 65.6 |

Sınıf başına duyarlılık (%):

| model                  |   nötr |   sakin |   mutlu |   üzgün |   öfkeli |   korkmuş |   tiksinmiş |   şaşkın |
|:-----------------------|-------:|--------:|--------:|--------:|---------:|----------:|------------:|---------:|
| baseline_bilstm        |   31.2 |    52.1 |    40.1 |    22.4 |     78.1 |      32.3 |        54.7 |     37   |
| wavlm_large_frozen_wls |   81.2 |    80.2 |    71.9 |    64.6 |     88   |      82.8 |        83.3 |     81.8 |
| wavlm_large_ft2        |   78.1 |    75.5 |    65.1 |    53.1 |     81.8 |      70.3 |        77.1 |     70.8 |