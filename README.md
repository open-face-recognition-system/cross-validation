# Vision Cross-Validation

K-fold evaluation (k = 5) of the face recognizers available to [vision-server](https://github.com/open-face-recognition-system/vision-server): Eigenfaces, Fisherfaces and LBPH. Each scenario trains on four folds of student photos and recognizes the fifth, reporting accuracy, weighted precision and weighted recall per algorithm.

Faces go through the same preprocessing as the server (grayscale, 220x220, histogram equalization), so the numbers describe what runs in production. The pip build of OpenCV used here is 5.x, while the server image ships Debian's 4.10; both expose the same `cv2.face` recognizers.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest            # sampling, recognition and the full pipeline on synthetic faces

DATABASE_URL=postgresql://vision:vision@localhost:5432/vision \
VISION_FILES_URL=http://localhost:3333/files \
.venv/bin/python download_photos.py --groups 35 15 5 --seed 42

.venv/bin/python validator.py --seed 42   # writes results/<date>-seed-42.txt
```

`download_photos.py` uses students with all 30 photos, picks each group at random with the given seed and writes them under `dataset/`, which git ignores. **Photos are biometric data**: only use them under the consent students gave and never commit them.

## Scenarios

| Test | Groups | Photos per student |
|---|---|---|
| `stratified` | 35, 15 and 5 students | 30, 15, 10 and 5, spread evenly across the five poses |
| `all` | every complete student | 30, folds drawn from the whole pool |
| `mask` | students photographed with masks | 30, 15, 10 and 5 |

## 2021 results

`results/2021-results.txt` holds the numbers from the original study, where LBPH came out ahead (96% to 99% accuracy). They were produced before a sampling bug was fixed: each fold took one photo per pose in directory order, so "5 photos per student" used a single pose, "10" and "15" only the first two or three poses, and "30" actually used 25 photos. Runs were also unseeded. Treat those numbers as indicative only and rerun the evaluation before drawing conclusions.

Evaluating YuNet + SFace against these recognizers is tracked in [#6](https://github.com/open-face-recognition-system/cross-validation/issues/6).
