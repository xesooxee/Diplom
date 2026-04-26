# Чихрийн шижин илрүүлэх систем — Код Review

> **Огноо:** 2026-04-15  
> **Хамрах хүрээ:** `main.py`, `model/train.py`, `model/XGBoost.py`, `model/logisticRegression.py`, `model/svm.py`, `model/evaluate.py`

---

## Ерөнхий үнэлгээ

Дипломын ажил бүтцийн хувьд **сайн зохион байгуулагдсан**, практик хэрэглээнд **бэлэн** шинжтэй байна. FastAPI backend, 4 загварын харьцуулалт, мэргэжлийн зөвлөмж гаргах функционал зэрэг нь ажлыг ердийн сургалтын төслөөс дээш түвшинд гаргаж байна.

---

## Давуу талууд

### 1. Архитектур
- `MODEL_REGISTRY` dict ашиглан загваруудыг нэг газраас удирдаж байгаа нь цэвэр шийдэл.
- `asynccontextmanager` + `asyncio.to_thread` ашиглан event loop-ийг блоклохгүйгээр загвар ачааллаж байгаа нь зөв.
- `Pipeline(scaler → clf)` хэлбэр ашигласан нь inference үед data leakage-аас сэргийлнэ.

### 2. Загварын сонголт
- 4 өөр алгоритм (RF, LR, SVM, XGBoost) харьцуулах нь дипломын шаардлагад тохирсон.
- XGBoost-д **Feature Engineering** (4 нэмэлт feature), **SMOTE**, **threshold tuning** ашигласан нь мэдлэгийн гүнийг харуулж байна.
- `class_weight="balanced"` тохиргоо бүх загварт хэрэглэсэн нь тэнцвэргүй датасеттэй зөв ажиллаж байгааг харуулна.

### 3. Validation
- Pydantic `field_validator` ашиглан input validation хийсэн — зөв хандлага.
- `PatientData`-д тоон утгуудын хязгаар (`ge`, `le`) тохируулсан нь API-ийг аюулгүй болгоно.
- 5-Fold Stratified CV ашигласан нь нийтлэг стандарт.

### 4. Evaluate скрипт
- 6 өөр диаграм (ROC, Bar, Confusion Matrix, Feature Importance, Radar, Table) автоматаар гаргаж байгаа нь дипломын тайланд шууд ашиглах боломжтой.

### 5. Зөвлөмж систем (`_recommend`)
- ADA 2024, WHO, CDC эх сурвалжид тулгуурласан нь эрдэм шинжилгээний хандлагыг харуулна.
- Утгын хэмжээнд тулгуурласан нарийвчилсан зөвлөмж (HbA1c ≥9.0, ≥6.5, ≥5.7) — нарийн боловсруулалт.

---

## Алдаа ба сул талууд

### Критик (засах шаардлагатай)

#### 1. `_encode()` дахь label encoding шинэ order-тэй зөрчилдөх эрсдэл
**Файл:** [main.py:145](main.py#L145)

```python
smoking_enc = SMOKING_CLASSES.index(data.smoking_history)
# SMOKING_CLASSES = ["No Info", "current", "ever", "former", "never", "not current"]
```

Гэтэл `train.py` болон `logisticRegression.py`-д `LabelEncoder().fit_transform()` ашигладаг бөгөөд энэ нь **алфавитын дарааллаар** кодлоно:
`["No Info", "current", "ever", "former", "never", "not current"]` → `[0, 1, 2, 3, 4, 5]`

Одоогоор санамсаргүйгээр таарч байж болох ч `fit_transform`-ын дараалал өөрчлөгдвөл **inference үр дүн буруу** болно.

**Засвар:** Сургалтын үед LabelEncoder-ийг хадгалж, inference үед ачаалах.

```python
# train.py - хадгалах
joblib.dump({"pipeline": pipe, "le_gender": le_gender, "le_smoking": le_smoking}, path)
```

---

#### 2. XGBoost pipeline.pkl нь dict, бусад нь sklearn Pipeline — тохируулгагүй
**Файл:** [main.py:333–348](main.py#L333)

```python
if isinstance(obj, dict):          # XGBoost
    ...
else:                               # RF, LR, SVM
    ...
```

`loaded_models[key]` нь загварын төрлөөс хамааран өөр бүтэцтэй байна. Шинэ загвар нэмэх үед энэ шалгалт хуучирна. `ModelWrapper` abstract class эсвэл callable protocol ашиглавал илүү найдвартай.

---

### Чухал (сайжруулах зүйтэй)

#### 3. `evaluate.py`-д train/test split давтагдаж байна
**Файл:** [model/evaluate.py:64–67](model/evaluate.py#L64)

```python
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

Энэ нь `train.py`-тай **ижил seed** ашигласан тул ижил test set гарч байна. Гэхдээ `evaluate.py` нь загварыг `.pkl`-аас ачааллаж үнэлдэг тул энэ нь зарчмын хувьд зөв. **Гэвч** XGBoost нь threshold tuning-д validation set ашигласан (`X_val`), тэр `X_val` нь энэ test set дотор байж болзошгүй. Энэ нь evaluate-ийн үнэлгээнд **бага зэрэг хэт өндөр** үзүүлэлт гарах шалтгаан болно.

**Засвар:** `X_train_fit` дотроос тусгаарлан, `X_test`-ийг огт хөндөхгүй байхыг XGBoost скриптэд баталгаажуулах.

---

#### 4. Хатуу кодлогдсон LOCAL хаяг
**Файл:** [model/train.py:5–6](model/train.py#L5) (docstring дотор)

```
cd /Users/temuulen/Documents/diplom/backend
```

Энэ нь таны машин дээрх замыг hardcode хийсэн байна. `BASE_DIR = pathlib.Path(__file__).resolve().parent.parent` ашигласан нь зөв, гэхдээ docstring-уудад хуучин зам үлдсэн байна.

---

#### 5. CORS тохиргоо — production-д аюулгүй биш
**Файл:** [main.py:75–88](main.py#L75)

```python
_origins = [..., "http://localhost:5173"]
app.add_middleware(CORSMiddleware, allow_origins=_origins, ...)
```

`ENV` environment variable-аар production/development ялгаж байгаа нь сайн. Гэвч `allow_credentials=True` нэмэх шаардлага гарвал `allow_origins=["*"]` ашиглах боломжгүй болохыг анхаарах хэрэгтэй.

---

#### 6. `_risk()` функцийн threshold нь тайлбаргүй
**Файл:** [main.py:173–178](main.py#L173)

```python
if probability < 0.3:   # → "Бага"
if probability < 0.6:   # → "Дунд"
else:                   # → "Өндөр"
```

`0.3`, `0.6` тооцоолол ямар эх сурвалжид тулгуурлав? ADA эсвэл бусад стандартад нийцүүлж, коммент нэмэх хэрэгтэй.

---

### Сайжруулалт (optional)

#### 7. `predict_sample()` дахь LabelEncoder давтагдаж байна
**Файл:** [model/train.py:183–185](model/train.py#L183), [logisticRegression.py:177–179](model/logisticRegression.py#L177), [svm.py:120–123](model/svm.py#L120)

3 файл бүрт ижил LabelEncoder-ийн хатуу кодлогдсон хэсэг байна. Нэг `utils.py`-д гаргаж авч болно.

---

#### 8. SVM hyperparameter tuning хийгдээгүй
**Файл:** [model/svm.py:37–45](model/svm.py#L37)

```python
SVM_PARAMS = {"kernel": "rbf", "C": 10.0, "gamma": "scale", ...}
```

RF болон XGBoost нь hyperparameter search хийдэг ч SVM нь `C=10.0` гэж хатуу тохируулсан. `GridSearchCV` эсвэл `RandomizedSearchCV` нэмэх нь загваруудын харьцуулалтыг илүү шударга болгоно.

---

#### 9. `/health` endpoint-д загварын version мэдээлэл байхгүй
**Файл:** [main.py:357–364](main.py#L357)

Сургасан огноо, датасетийн хэмжээ, загварын version зэргийг `/health`-д нэмвэл production monitoring-д тустай.

---

#### 10. `metabolic_score` feature-ийн scale асуудал
**Файл:** [main.py:164](main.py#L164), [model/XGBoost.py:58](model/XGBoost.py#L58)

```python
metabolic_score = bmi + hba1c * 10 + blood_glucose_level / 10
```

`bmi` (~18–40), `hba1c*10` (~50–90), `gluc/10` (~10–50) — нийлвэр нь хувийн жинг тэнцүү тооцоогүй тул энэ feature-ийн эклектик шинж нь загварын тайлбарлагдах байдлыг хүндэлнэ. Дипломд энэ оновчтой байдлыг тайлбарлах шаардлагатай.

---

## Бүтцийн зөвлөмж

```
diplom/
├── data/
│   └── diabetes.csv
├── model/
│   ├── train.py            ✓ RF
│   ├── XGBoost.py          ✓ XGB
│   ├── logisticRegression.py ✓ LR
│   ├── svm.py              ✓ SVM
│   ├── evaluate.py         ✓
│   └── utils.py            ← БАЙХГҮЙ (давтагдах кодыг энд нэгтгэх)
├── main.py                 ✓ FastAPI
└── requirements.txt        ✓
```

`utils.py` нэмж дараах функцуудыг нэгтгэх:
- `encode_categorical(df, col, classes)` — LabelEncoder хадгалах/ачаалах
- `risk_level(prob)` — хэрэв `main.py`-д байгааг тест хийх шаардлага гарвал

---

## Дүгнэлт

| Хэсэг | Үнэлгээ | Тайлбар |
|---|---|---|
| ML Pipeline | ★★★★☆ | SMOTE, threshold tuning, CV — сайн |
| FastAPI API | ★★★★☆ | Async loading, Pydantic validation — зөв |
| Загварын харьцуулалт | ★★★★☆ | 4 загвар, 6 диаграм — дипломд тохирсон |
| Зөвлөмж систем | ★★★★★ | ADA/WHO стандарт — маш сайн |
| Кодын цэвэр байдал | ★★★☆☆ | LabelEncoder давтагдах, dict/Pipeline зөрүү |
| Аюулгүй байдал | ★★★★☆ | Input validation сайн, CORS тохиромжтой |

**Ерөнхий:** Дипломын ажилд зориулсан **A-** түвшний ажил. Дээрх 2 критик асуудлыг засвал production-д ашиглахад бэлэн болно.
