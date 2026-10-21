# Araştırma Raporu: Futbol Maç Sonucu Tahmini ve Bahis Piyasası

## Özet

Herkese açık verilerle (maç sonuçları, xG, kadrolar, bahis oranları) futbol maç sonuçlarını tahmin eden modeller kuruldu ve bahis piyasasıyla karşılaştırıldı. **22 ligde ve ~44 bin maçta, ne modeller ne kural tabanlı stratejiler ne de piyasayı girdi olarak kullanan beraberlik sıralaması, piyasaya karşı sistematik bir avantaj sağladı.**

- Model, kendi içinde belirgin şekilde gelişti: Premier League 1X2 RPS 0.2027'den 0.1987'ye indi. xG ve kadro verisinin gerçek katkısı var.
- Kapanış oranları yine de daha isabetli (RPS 0.1935). Model ile piyasa birleştirildiğinde modellere kalan ağırlık 0.
- Önceden kayıt altına alınan başarı ölçütlerinin hiçbiri sağlanmadı.
- Beraberliğe en yatkın maçlar doğru bulunuyor (isabet %32–33, genel oran %26), ama piyasa bunu zaten fiyatlıyor. Açılış oranından bahis ROI'si negatif.

## Yöntem

**Veri kaynakları**

| Kaynak | İçerik |
|---|---|
| football-data.co.uk | Sonuçlar; açılış ve kapanış oranları (1X2, 2.5 üst/alt); 8 bahisçinin kapanış oranları; yaklaşan fikstürler |
| Understat | Maç bazında xG (5 büyük lig); maç başına kadrolar ve oyuncu xG/xA |

**Modeller**

| Kod | Model |
|---|---|
| `elo` | Ev sahibi avantajı, gol farkı çarpanı ve sezon başı geri çekmeli Elo; olasılıklar ordered logit ile |
| `dc` | Zaman ağırlıklı Dixon-Coles (analitik gradyan) |
| `dcx` | Gol yerine 0.7·xG + 0.3·gol ile beslenen Dixon-Coles |
| `dcl` | Kadro düzeltmeli `dcx`: oyuncu (xG+xA)/90 puanlarından "sahadaki 11 − tipik 11" farkı, katsayı walk-forward öğreniliyor |
| `blend` | Modellerin walk-forward öğrenilen doğrusal birleşimi |
| `mkt` / `open` / `cons` | Payı temizlenmiş (Shin) kapanış / açılış / çoklu bahisçi konsensüsü |
| `stack` / `stack_open` | Modeller + kapanış / açılış piyasasının doğrusal birleşimi |

**Değerlendirme ilkeleri**

- **Walk-forward:** Her tahmin yalnızca maçtan önce bilinen veriyle yapıldı; modeller haftalık yeniden eğitildi. Geçmiş hafta modunda tüm modeller `as_of` tarihinde donduruldu. Sızıntı testi: test penceresindeki skorlar, xG ve kapanış oranları bozulduğunda tahminler birebir aynı kaldı.
- **Metrikler:** RPS, log loss, Brier, kalibrasyon. Bahis tarafında CLV (açılış fiyatının payı temizlenmiş kapanış olasılığına göre değeri); ROI standart hatasıyla birlikte raporlandı.
- **Ön kayıt:** Başarı ölçütleri ve strateji kuralları sonuçlar görülmeden yazıldı (`strategies/rules.toml`, commit 02dc258).
- **Keşif / doğrulama:** Son 2 sezon doğrulama olarak ayrıldı; ayarlar liglere göre değiştirilmedi.

## Bulgular

### 1. Premier League, 1X2 (2020/21–2025/26, 1.900 ortak maç)

| Model | RPS | Log loss |
|---|---|---|
| elo | 0.2006 | 0.9789 |
| dc | 0.2017 | 0.9830 |
| dcx | 0.2000 | 0.9760 |
| dcl | 0.1994 | 0.9748 |
| **blend** | **0.1987** | **0.9725** |
| open (açılış) | 0.1948 | 0.9591 |
| cons (konsensüs) | 0.1935 | 0.9553 |
| **mkt (Pinnacle kapanış)** | **0.1935** | **0.9551** |
| stack | 0.1936 | 0.9554 |

- Birleşimde ağırlıklar: `blend` için elo 0.35, dcl 0.65. `stack` ve `stack_open` için modellere 0, piyasaya 1.
- Kadro etkisi β = 0.19–0.22: kadro zayıfladıkça gol beklentisi beklenen yönde düşüyor.
- Açılış oranından bahis CLV'si: ens −%4.8, blend −%4.8, stack_open −%4.0.
- Hiperparametre ayarı, ayara girmemiş son iki sezonda kazanç sağlamadı (−0.0003 ile +0.0008 RPS). Varsayılanlar korundu.

### 2. Premier League, 2.5 üst/alt

| Model | Brier | Log loss |
|---|---|---|
| dc25 | 0.4916 | 0.6851 |
| dcx25 | 0.4829 | 0.6759 |
| open25 | 0.4813 | 0.6741 |
| mkt25 | 0.4810 | 0.6738 |

`stack25` birleşiminde xG modelinin ağırlığı 0.24. Açılış oranından bahis CLV'si −%3.5.

### 3. Beş büyük lig (aynı ayarlar)

| Lig | RPS farkı (model − piyasa) | Brier farkı (2.5) | Modele ağırlık 1X2 | Modele ağırlık 2.5 | CLV 2.5 |
|---|---|---|---|---|---|
| E0 | 0.0056 | 0.0020 | 0.00 | 0.24 | −%2.5 |
| SP1 | 0.0042 | 0.0099 | 0.00 | 0.01 | – |
| I1 | 0.0050 | 0.0094 | 0.00 | 0.00 | −%1.3 |
| D1 | 0.0048 | 0.0115 | 0.00 | 0.00 | −%3.8 |
| F1 | 0.0048 | 0.0129 | 0.00 | 0.00 | −%3.4 |

**Ön kayıtlı ölçüt** ("en az 4 ligde 2.5 için modele 0.2'nin üzerinde ağırlık ve pozitif CLV"): sağlanmadı.

### 4. Alt ve küçük ligler (12 lig, xG'siz)

Model ile piyasa arasındaki fark tüm liglerde büyük liglerle aynı düzeyde ya da daha büyük: RPS farkı 0.004–0.010, 2.5 Brier farkı 0.009–0.030. Hiçbir ligde modele anlamlı ağırlık kalmadı.

**Ön kayıtlı ölçüt** ("en az 3 ligde ağırlık 0.2'nin üzerinde ve pozitif CLV"): sağlanmadı. "Alt liglerde piyasa daha verimsizdir" hipotezi bu veride doğrulanmadı.

### 5. Ön kayıtlı strateji kuralları (5 büyük lig, doğrulama dönemi)

| Kural | Bahis | İsabet | Piyasa beklentisi | ROI | CLV |
|---|---|---|---|---|---|
| Kontrol: tüm beraberlikler | 3503 | %25.2 | %24.5 | −%4.0 | −%4.7 |
| Düşük toplam gol + dengeli → beraberlik | 140 | %27.9 | %31.1 | −%14.2 | −%2.0 |
| Piyasa az gol ve dengeli bekliyor → beraberlik | 368 | %32.3 | %30.8 | +%0.2 | −%3.0 |
| Model beraberliği piyasadan 3+ puan yüksek görüyor | 272 | %14.7 | %17.7 | −%26.5 | −%6.2 |
| Düşük toplam + model farkı → beraberlik | 68 | %19.1 | %24.0 | −%20.7 | −%3.9 |
| Kontrol: tüm altlar | 3485 | %46.9 | %47.0 | −%5.5 | −%5.1 |
| Düşük xG → alt | 224 | %62.1 | %62.2 | −%4.1 | −%2.3 |
| Model altı piyasadan 4+ puan yüksek görüyor | 297 | %45.8 | %41.9 | +%1.6 | −%5.8 |

- 8 kuralın hiçbiri "umut verici" çıkmadı (ölçüt: doğrulamada pozitif CLV ve p < 0.0063).
- Az gollü maçlarda beraberlik ve alt gerçekten daha sık geliyor, ama piyasa olasılığı bunu zaten yansıtıyor.
- Modelin piyasadan ayrıştığı yerlerde haklı çıkan çoğunlukla piyasa.

### 6. Beraberlik sıralaması (22 lig, 44.163 maç, 254 hafta)

Her hafta tüm liglerden beraberlik olasılığı en yüksek N maç seçildi.

| Liste | Sıralama | İsabet (tümü) | İsabet (doğrulama) | ROI açılış (doğrulama) |
|---|---|---|---|---|
| 3 | birleşik | %33.5 | %28.9 | −%15.8 |
| 3 | yalnız piyasa | %34.0 | %30.0 | −%13.0 |
| 5 | birleşik | %32.8 | %31.1 | −%8.1 |
| 5 | yalnız piyasa | %33.7 | %31.3 | −%7.6 |
| 7 | birleşik | %32.6 | %31.9 | −%4.8 |
| 7 | yalnız piyasa | %33.4 | %31.9 | −%4.8 |
| Tüm maçlar | – | %26.2 | %26.3 | – |

- Birleşik modelin standartlaştırılmış ağırlıkları: piyasa +0.315, diğer tüm girdiler ~0.
- Gol beklentisi, güç farkı, alt olasılığı ve lig beraberlik oranı, piyasa olasılığının üzerine bilgi eklemiyor.
- Geçmiş hafta testi (12–19 Ekim 2026, 194 maç): 7 maçlık listede 1 beraberlik (beklenen 2.2); yalnız piyasa listesinde de 1/7. Tek hafta için şans sınırları içinde.

## Sonuç

Bahis oranları, bu projede kullanılabilen herkese açık bilginin tamamını zaten içeriyor ve üstüne bahisçi payı ekleniyor. Bu yüzden bu yöntemlerle düzenli bahis, beklenen değeri negatif bir iş.

Projenin kalıcı değeri:

- **Doğru kalibre edilmiş olasılıklar:** 1X2, 2.5 üst/alt, karşılıklı gol ve skor dağılımı.
- **Analiz raporu:** Yaklaşan maçlar için HTML rapor (`predict`) ve beraberliğe en yatkın maçların listesi (`draws list`).
- **Sızıntısız araştırma altyapısı:** Yeni bir hipotez aynı düzenekte dakikalar içinde test edilebilir (`backtest`, `compare`, `strategy`, `draws backtest`).

## Sınırlamalar

- **"Açılış" oranı:** football-data'nın açılış oranı, maçtan birkaç gün önce toplanan orandır; gerçek ilk fiyat değildir.
- **Kapanış oranı:** 2019/20 öncesi sezonlarda yok; bu sezonlar yalnızca ısınma için kullanıldı.
- **Kapsam:** xG ve kadro verisi yalnızca 5 büyük ligde var. Kadro modeli yalnızca Premier League'de test edildi. Canlı kadro verisi olmadığı için ileriye dönük tahminlerde kadro düzeltmesi kullanılmıyor.
- **Hafta içi güncelleme:** `draws backtest`'te Elo ve lig beraberlik oranı hafta içinde güncelleniyor. "Liste Pazartesi hazırlandı" senaryosuna göre bu küçük bir iyimserlik payı yaratır; `draws list` ise her şeyi `as_of` tarihinde donduruyor.
- **Kapsam dışı:** Canlı (maç içi) oranlar, korner/kart pazarları, borsa (exchange) fiyatları ve oranların bahisçiler arasında yayılma hızı denenmedi.

## Yeniden üretme

```bash
pip install -e .
sportsforecast check-sources
sportsforecast fetch-lineups --league E0 --seasons 1819-2526      # ~50 dk, bir kez
sportsforecast backtest --league E0 --seasons 1819-2526 --lineups
sportsforecast tune --league E0 --seasons 1819-2526
sportsforecast compare --leagues E0,SP1,I1,D1,F1 --seasons 1819-2526
sportsforecast compare --leagues E1,E2,E3,SC0,T1,N1,B1,P1,SP2,I2,D2,F2 --seasons 1819-2526 --no-xg
sportsforecast strategy --leagues E0,SP1,I1,D1,F1 --seasons 1819-2526
sportsforecast draws backtest --seasons 1819-2526
sportsforecast draws list --as-of 2026-10-12 --days 8 --top 7
sportsforecast predict --league E0
```

Sonuçlar 21 Ekim 2026 itibarıyla indirilen veriyle üretildi. Mevcut sezon dosyaları güncellendikçe sayılar küçük ölçüde değişebilir.
