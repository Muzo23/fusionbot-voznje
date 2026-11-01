# Korak 15b – sustavno podešavanje šuma EKF-a i COV_INFLATE (NEES + L1, Monte Carlo)

**Zahtjev zadatka:** precizna procjena stanja u uvjetima nesigurnosti (konzistentnost filtra, NEES);
fuzija podataka iz više izvora; predmet Evolucijska robotika (evolucijski optimizator – diferencijalna evolucija).

## Zašto

Parametri šuma u koraku 15 bili su početne procjene iz izmjerenih izvora pogreške. Kod podešen „na oko” ne
jamči da kovarijanca filtra odgovara stvarnoj pogrešci. Mjera je NEES (Bar-Shalom i sur., 2001):
za konzistentan filtar srednja vrijednost je 3 za (x, y, θ), 2 za (x, y) i 1 za θ. Prije 15b:
A 0,6–1,0 (preoprezan), C 1,1–1,7, B 0,7–0,8, D 40–94 (izrazito samouvjeren), E 0,8.

## Infrastruktura: replay estimatora (`experiment/replay.py`)

* Navigacija (FSM, regulatori) radi samo na enkoderima i senzorima blizine. Procjena poze ulazi u vožnju
  samo kroz topološku kartu (pridruživanje raskrižja) i traženje cilja. Dok je karta točna, putanja i sva
  mjerenja ne ovise o estimatoru, pa se snimljena vožnja može ponovno provući kroz bilo koji estimator.
* Snimke: konfiguracija E (karta uvijek točna), sve ulaze ima recorder (enkoderi, žiroskop, senzori blizine
  sa šumom, LiDAR, događaji raskrižja). Dodano u 15b: kutovi zraka LiDAR-a spremaju se kao float64, a
  početna kalibracija biasa u radijanima (`summary.json`).
* **Provjera (infrastruktura):** replay iste konfiguracije s istim parametrima reproducira snimljenu procjenu
  na 10⁻¹² mm; replay D nad snimkom E daje točno rezultat prave D vožnje (T1 s1: L1 63,17 mm, NEES 35,50).
  Prije ispravka (float32 kutovi) odstupanje je raslo do 10 mm – zatvorena petlja karta ↔ procjena
  pojačava i najmanju numeričku razliku.
* Jedna evaluacija: A/B/C ~3 s, D/E ~1 min (umjesto 2–5 min za cijelu vožnju).

Snimke: `python -m experiment.batch --config E --noise realistic --mazes T1 T2 T3 M1 M2 M3 M5 --seeds 1-3 --tag rec15b --extra --offline-lidar`
(21/21 uspjeh, sve karte točne).

**Skup za podešavanje:** T1, T3, M2, M5 × s1–2 (8). **Validacijski skup:** T2, M1, M3 × s1–3 (9) – optimizator ga ne vidi.

## Metoda (`experiment/tune.py`)

* Optimizator: diferencijalna evolucija (Storn i Price, 1997; `scipy.optimize.differential_evolution`),
  parametri u log10 prostoru, granice ±1,5 dekade oko početne vrijednosti, populacija 6–8 × broj parametara,
  10 generacija, latin hypercube inicijalizacija, sjeme 15.
* Funkcija cilja (po vožnji, srednja vrijednost):
  J = ln(L1 / L1_ref) + ln(NEES_xy / 2)² + ln(NEES_θ / 1)²  (L1_ref = L1 s početnim parametrima).
  Kvadrat logaritma kažnjava jednako samouvjerenost i preopreznost (faktor 2 gore = faktor 2 dolje).
* Faze, svaka fiksira rezultat prethodne:
  1. `enc` – A (samo enkoderi): Q_DS_PER_M, Q_TH_ENC_PER_RAD, Q_TH_ENC_PER_M (L1 ne ovisi o Q – čista konzistentnost).
  2. `gyro` – C: Q_TH_GYRO_PER_RAD.
  3. D: COV_INFLATE – sken (D je skup za DE na 2 jezgre; jedna dimenzija dovoljna).

```
cd fusionbot\src
python -m experiment.tune enc  --tag rec15b --generations 10
python -m experiment.tune gyro --tag rec15b --generations 10 --popsize 8 --fix ../results/tuning/enc.json
python -m experiment.tune scan --config D --param scan_match.COV_INFLATE --values 10 30 100 300 1000 3000 --fix ../results/tuning/enc.json ../results/tuning/gyro.json
python -m experiment.tune table --fix ../results/tuning/enc.json ../results/tuning/gyro.json --set scan_match.COV_INFLATE=100 --out table_new.json
```

## Rezultati podešavanja

| parametar | prije (15) | poslije (15b) | fizikalna provjera |
|---|---:|---:|---|
| Q_DS_PER_M (put) | 10 mm/√m | **2,0 mm/√m** | 1024 CPR, bez proklizavanja offline |
| Q_TH_ENC_PER_RAD (okret) | 1° / 90° | **0,71° / 90°** | izmjereni gubitak uživo 0,86 / 0,97° |
| Q_TH_ENC_PER_M (ravno) | 0,3° / √m | **0,056° / √m** | donja granica pretrage – zanemarivo |
| Q_TH_GYRO_PER_RAD | 0,30° / 90° | **0,26° / 90°** | skala MPU-6050 σ 0,3 % → 0,27° / 90° |
| GYRO_BIAS_RW | 0,002 °/s/√s | nepromijenjeno | vrijednost iz tablice podataka |
| COV_INFLATE (D) | 1000 | **100** | vidi niže |

**Odbačena varijanta (gyro + GYRO_BIAS_RW):** DE je povećao pomak biasa 3,8× iznad tablice podataka i tako
kupio NEES uz +6 % L1 (validacija +10 %): parametar mijenja pojačanje ZUPT-a, dakle i samu procjenu.
Pomak biasa je poznato fizikalno svojstvo (profil šuma koristi upravo tu vrijednost) i ne podešava se
(`results/tuning/gyro_with_bias_rw_rejected.json`).

### Replay, prije → poslije (srednje vrijednosti; L1 RMSE, ANEES za (x, y, θ))

| | skup | L1 prije | L1 poslije | ANEES prije | ANEES poslije | NEES_xy / NEES_θ poslije |
|---|---|---:|---:|---:|---:|---|
| A | podešavanje | 481 mm | 481 mm | 1,02 | **2,36** | 1,43 / 1,72 |
| A | validacija | 183 mm | 183 mm | 0,60 | **1,47** | 0,98 / 0,86 |
| B | podešavanje | 3,3 mm | 3,0 mm | 0,69 | **1,04** | 0,42 / 0,65 |
| B | validacija | 3,3 mm | 3,1 mm | 0,77 | **1,18** | 0,47 / 0,77 |
| C | podešavanje | 182 mm | 182 mm | 1,70 | **2,39** | 1,47 / 2,02 |
| C | validacija | 86 mm | 86 mm | 1,09 | **1,57** | 0,96 / 1,34 |
| D | podešavanje | 93 mm | **70 mm** | 94 | 622 | 543 / 14,5 |
| D | validacija | 60 mm | **48 mm** | 40 | 234 | 200 / 10,0 |
| E | podešavanje | 3,1 mm | 2,9 mm | 0,82 | **1,79** | 1,25 / 0,65 |
| E | validacija | 3,0 mm | 3,2 mm | 0,81 | **2,04** | 1,43 / 0,68 |

A, B, C i E su sada blizu konzistentnosti (ANEES 1–2,4, uz 95 %–100 % uzoraka unutar granice 95 %) i ostaju
na opreznoj strani, što je za navigaciju sigurnije od samouvjerenosti. Validacijski skup pokazuje isti
smjer, dakle rezultat nije prilagodba skupu za podešavanje. Točnost A, B, C i E je nepromijenjena (A i C
nemaju korekcija položaja; B i E su na ~3 mm).

## D: zašto COV_INFLATE ne može učiniti D konzistentnim

Sken s podešenim Q (21 vožnja, srednje vrijednosti):

| COV_INFLATE | 10 | 30 | **100** | 300 | 1000 | 3000 |
|---|---:|---:|---:|---:|---:|---:|
| L1 [mm] | 132 | 84 | **62** | 76 | 113 | 180 |
| ANEES | 11 101 | 2 233 | 481 | 317 | 240 | 297 |

* ANEES ne pada ispod ~140 ni za jednu vrijednost (D je 50–200× samouvjereniji nego što bi smio biti).
  Uzrok je strukturni: sken se poravnava s kartom izgrađenom iz **iste** procjene, pa mjerenje i stanje
  nisu neovisni, a EKF ih tretira kao neovisne. Kovarijanca kolabira pri svakom skenu (10 Hz), a pogreška
  se akumulira s kartom. To je poznata nekonzistentnost EKF-SLAM-a bez praćenja korelacija karte i poze
  (Julier i Uhlmann, 2001; Castellanos i sur., 2004). Pravi lijek je zajednička procjena karte i poze
  (EKF-SLAM s orijentirima ili graf-SLAM sa zatvaranjem petlje), što je izvan opsega rada.
* Stari veliki Q (10 mm/√m) djelomično je skrivao samouvjerenost: uz COV_INFLATE = 1000 ANEES je bio 40–94.
  S konzistentnim Q-om to više nije moguće, a COV_INFLATE samo zamjenjuje točnost za (i dalje neupotrebljiv) NEES.
* **Odluka:** COV_INFLATE = 100, najtočnija vrijednost. D je točniji nego prije (validacija 60 → 48 mm,
  podešavanje 93 → 70 mm), a nekonzistentnost D-a navodi se u radu kao rezultat i ograničenje (motivacija za E).
  Napomena: unaprijed definirana J (s članom NEES) bi zbog velikih iznosa NEES-a birala 1000. NEES član je
  smislen samo blizu konzistentnosti; ovdje nijedna vrijednost nije unutar reda veličine od 3, pa se bira
  prema točnosti. **Ovo je odluka za potvrdu.**
* **Isprobano i odbačeno – Covariance Intersection** (`SCAN_MODE = "ci"`): R = R_poravnanja + varijanca
  poze s kojom su zidovi ucrtani (`OccupancyGrid.s2_xy/s2_th`), fuzija preko CI (Julier i Uhlmann, 1997),
  koja je konzistentna za bilo kakvu korelaciju. Rezultat (T1, M2, M5 s1): ANEES 3,04 (konzistentno), ali
  L1 0,5–1,4 m – lošije i od A. Slabe korekcije puštaju da se karta izobliči, a poravnanje s izobličenom
  kartom vuče procjenu dalje. Kod ostaje kao opcija (kao i „split” iz koraka 17), zadano je `plain`.

## E

E (sve) je i nakon 15b najtočniji (~3 mm) i sada konzistentan (ANEES 1,8–2,0 umjesto 0,8): PROX-LOC
(mreža labirinta) daje apsolutnu informaciju i drži kovarijancu poštenom unatoč samouvjerenom LiDAR-u.

## Provjera u zatvorenoj petlji (prave offline vožnje s novim zadanim vrijednostima)

Replay pretpostavlja točnu kartu. Navigacija ne koristi kovarijancu, pa se A i C s novim Q voze identično.
Za B, D i E (gate ovisi o P) i za dinamičke prepreke (gating uz prepreke) napravljene su prave vožnje:

```
python -m experiment.batch --config B D E --noise realistic --mazes T1 T2 T3 M1 M2 M3 M5 --seeds 1 --tag tuned15b --extra --offline-lidar
python -m experiment.batch --config A B --noise realistic --mazes M2 M3 M4 M5 --seeds 1-2 --dyn ALL --tag dyn15b
```

| | vožnje | S4 | karta točna | L1 RMSE | najveća pogreška u događaju | ANEES | K2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 7 | 7/7 | 7/7 | 3,0 mm | 18 mm | 1,10 | 89,0 % |
| D | 7 | 7/7 | 7/7 | 61,3 mm | 223 mm | 445 | 85,7 % |
| E | 7 | 7/7 | 7/7 | 3,1 mm | 15 mm | 1,97 | 90,0 % |

Dinamika `ALL_dyn15b` (M2–M5 × 2), usporedba s `ALL_dyn16`:

| | S4 | karta točna | najveća pogreška u događaju | ANEES | zaustavljanja / lažna |
|---|---:|---:|---:|---:|---:|
| A | 6/8 (prije 6/8) | 4/8 (prije 4/8) | 1892 mm | 2,00 | 14 / 0 |
| B | 8/8 (prije 8/8) | 8/8 (prije 8/8) | **17 mm** (prije 19 mm) | 1,26 | 18 / 0 |

Zatvorena petlja daje iste brojke kao replay (B 3,0 mm, D 61 mm, E 3,1 mm), dakle replay je valjan alat za
podešavanje. Tješnji gate zbog manjeg Q-a ne šteti: B uz prepreke ima istu uspješnost i manju najveću pogrešku,
a gate odbacuje mjerenja blizu prepreka (1–33 po vožnji) kao i prije. A M4 ne uspijeva kao i u 14b
(`goal_not_in_map` – drift odometrije, ne ovisi o Q).

## Testovi

`tests/test_step15b_tuning.py` (6): točnost replaya na kratkoj vožnji E, vraćanje parametara, funkcija cilja
(simetrična, 0 za konzistentan filtar), CI ne kolabira pri ponovljenoj fuziji koreliranih podataka i ne
dira bias, fizikalna provjera podešenih vrijednosti. Ukupno 212 offline testova, svi prolaze.

## Za rad

* Poglavlje 3 (metoda): replay kao alat za Monte Carlo evaluaciju estimatora; DE kao evolucijski optimizator
  parametara šuma (Storn i Price 1997); funkcija cilja; podjela podešavanje / validacija.
* Poglavlje 4 (rezultati): tablica prije/poslije; sken COV_INFLATE (slika L1 i ANEES u ovisnosti o
  COV_INFLATE, log-os); D kao primjer strukturne nekonzistentnosti, E kao rješenje fuzijom.
* Ograničenje: Q je podešen offline (simulirani pogon bez proklizavanja); uživo (live17) treba provjeriti NEES.
  Ako je uživo NEES A/C bitno > 3, Q_DS i Q_TH_ENC_PER_M povećati prema uživo izmjerenim vrijednostima.

## Podaci

`docs/korak15b_data/` (kopija iz `results/`, koji nije u gitu): `enc.json`, `gyro.json` (najbolje vrijednosti +
cijela povijest DE), `gyro_with_bias_rw_rejected.json`, `table_old.json` / `table_new.json` /
`table_new_D300.json` (replay po vožnjama), `scan_D_*.json` (sken COV_INFLATE; za 1000 i 3000 s podešenim Q
sačuvane su samo srednje vrijednosti u tablici gore), CSV-ovi batcheva `rec15b`, `tuned15b`, `ALL_dyn15b`.
