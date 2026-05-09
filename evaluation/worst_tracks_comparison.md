# Worst Performing Tracks: Beats & Downbeats

This document compiles the absolute hardest tracks in the Jazz Trio Database (JTD) for beat and downbeat tracking, comparing the **Fine-tuned Beat This!**, **Baseline Beat This!**, and **Madmom** models.

The tables include any track that appeared in the **10 worst** list for either model, with missing metrics populated from the raw [split_comparison_track_level.csv](file:///Users/jonathandavid/NYU/Spring%202026/MIR/MIR-Jazz-Beat-Tracking/evaluation/csvs/split_comparison_track_level.csv) file.

---

## 1. Worst Tracks for Beat F-measure

Values in **bold** represent a track that fell into that specific model's 10 worst list.

| Artist - Song (Year) | Track ID | Fine-Tuned F | Beat This! F | Madmom F |
| :--- | :--- | :---: | :---: | :---: |
| **Keith Jarrett** - *I Thought About You* (1993) | `jarrettk-ithoughtaboutyou-peacockgdejohnettej-1993-340a16bd` | **0.6190** | **0.3750** | **0.4121** |
| **McCoy Tyner** - *Darn That Dream* (1989) | `tynerm-darnthatdream-sharpeascotta-1989-6cc187a8` | **0.5308** | **0.4682** | **0.5038** |
| **Chick Corea** - *Sophisticated Lady* (1989) | `coreac-sophisticatedlady-patituccijweckld-1989-85cb4d64` | **0.8315** | **0.5706** | 0.8142 |
| **Paul Bley** - *Syndrome* (1963) | `bleyp-syndrome-swallowslap-1963-e13458a9` | **0.7075** | **0.6192** | 0.8040 |
| **Hank Jones** - *Like Someone In Love* (1977) | `jonesh-likesomeoneinlove-duviviergjacksono-1977-75081d2b` | 0.9277 | **0.6270** | 0.9191 |
| **Keith Jarrett** - *Lisbon Stomp* (1968) | `jarrettk-lisbonstomp-hadencmotianp-1968-8994c848` | **0.8783** | **0.6540** | **0.5707** |
| **McCoy Tyner** - *The Wise One* (1991) | `tynerm-thewiseone-sharpeascotta-1991-a8ff2e12` | 0.9842 | **0.6602** | 1.0000 |
| **Bud Powell** - *Salt Peanuts* (1956) | `powellb-saltpeanuts-duviviergtaylora-1956-4acaf193` | 1.0000 | **0.6622** | 1.0000 |
| **Oscar Peterson** - *Woody'n You* (1961) | `petersono-woodynyou-brownrthigpene-1961-f34313d2` | 1.0000 | **0.6658** | 1.0000 |
| **Bobby Timmons** - *Walkin' Wadin' Sittin' Ridin'* (1964) | `timmonsb-walkinwadinsittinridin-jonesslucasr-1964-67413066` | 0.9892 | **0.6667** | 0.6667 |
| **Thelonious Monk** - *April In Paris* (1949) | `monkt-aprilinparis-rameygblakeya-1949-c50b1c36` | **0.6667** | 0.6667 | 0.6667 |
| **Ray Brown** - *My Heart Stood Still* (1978) | `brownr-myheartstoodstill-smithpbellsonl-1978-d3d8fdc4` | **0.6667** | 0.6667 | 0.6667 |
| **Ahmad Jamal** - *You Go To My Head* (1960) | `jamala-yougotomyhead-crosbyifournierv-1960-5a0737d3` | **0.7640** | 0.7606 | 0.7674 |
| **Kenny Barron** - *There Is No Greater Love* (1984) | `barronk-thereisnogreaterlove-williamsbrileyb-1984-90d41f7e` | **0.8726** | 0.8743 | 0.8293 |
| **Ahmad Jamal** - *Angel Eyes* (1961) | `jamala-angeleyes-crosbyifournierv-1961-69a58683` | **0.8947** | 0.8889 | 0.8947 |
| **Red Garland** - *Solar* (1977) | `garlandr-solar-carterrjoep-1977-a821803c` | 0.9811 | 1.0000 | **0.5785** |
| **Bill Evans** - *One For Helen* (1967) | `evansb-oneforhelen-gomezedejohnettej-1967-94d467a6` | 0.9569 | 0.8774 | **0.6105** |
| **Bill Evans** - *My Romance* (1980) | `evansb-myromance-johnsonmlabarberaj-1980-0ac208d7` | 0.9727 | 0.9954 | **0.6265** |
| **Bill Evans** - *Make Someone Happy* (1966) | `evansb-makesomeonehappy-israelscwisea-1966-1f814ca8` | 0.9631 | 0.9002 | **0.6278** |
| **Keith Jarrett** - *On Green Dolphin Street* (2001) | `jarrettk-greendolphinstreet-peacockgdejohnettej-2001-85ffc93e` | 0.9521 | 0.8540 | **0.6334** |
| **Bill Evans** - *My Romance* (1980) | `evansb-myromance-johnsonmlabarberaj-1980-b98b5cbc` | 0.9802 | 0.7976 | **0.6358** |
| **Chick Corea** - *Autumn Leaves* (1989) | `coreac-autumnleaves-patituccijweckld-1989-9f758273` | 0.9565 | 0.8427 | **0.6376** |

---

## 2. Worst Tracks for Downbeat F-measure

Values in **bold** represent a track that fell into that specific model's 10 worst list.

| Artist - Song (Year) | Track ID | Fine-Tuned F | Baseline F | Madmom F |
| :--- | :--- | :---: | :---: | :---: |
| **Bud Powell** - *Blues In The Closet* (1960) | `powellb-bluesinthecloset-pettifordoclarkek-1960-cf0f2bb4` | **0.0000** | **0.0000** | 0.0000 |
| **Dave Holland** - *Interface* (1989) | `hollandd-interface-joneshhigginsb-1989-56f4aaf0` | **0.0559** | 0.0000 | 1.0000 |
| **Oscar Peterson** - *I Love You* (1961) | `petersono-iloveyou-brownrthigpene-1961-5c99390e` | **0.0993** | **0.0000** | 0.0000 |
| **Cedar Walton** - *The Newest Blues* (1992) | `waltonc-thenewestblues-williamsdhigginsb-1992-e10c3edf` | **0.1511** | 0.0000 | 1.0000 |
| **Dave Holland** - *Trane Connections* (1989) | `hollandd-traneconnections-joneshhigginsb-1989-a13fb393` | **0.1637** | 0.0000 | 0.0000 |
| **McCoy Tyner** - *Darn That Dream* (1989) | `tynerm-darnthatdream-sharpeascotta-1989-6cc187a8` | **0.1852** | 0.1250 | 0.1527 |
| **Ahmad Jamal** - *On Green Dolphin Street* (1959) | `jamala-ongreendolphinstreet-crosbyifournierv-1959-8ff3c6f7` | **0.2032** | 0.0145 | 1.0000 |
| **Keith Jarrett** - *I Thought About You* (1993) | `jarrettk-ithoughtaboutyou-peacockgdejohnettej-1993-340a16bd` | **0.2062** | 0.2692 | 0.1428 |
| **Bill Evans** - *Turn Out The Stars* (1974) | `evansb-turnoutthestars-gomezemorellm-1974-c60c9a34` | **0.2370** | 0.0000 | 1.0000 |
| **Bill Evans** - *Minority* (1958) | `evansb-minority-jonessjoep-1958-687403fa` | **0.2613** | 0.0000 | 0.0000 |
| **Ahmad Jamal** - *Darn That Dream* (1959) | `jamala-darnthatdream-crosbyifournierv-1959-0abd7fe6` | 0.4536 | **0.0000** | **0.0000** |
| **Bud Powell** - *Salt Peanuts* (1956) | `powellb-saltpeanuts-duviviergtaylora-1956-4acaf193` | 0.4000 | **0.0000** | 1.0000 |
| **Wynton Kelly** - *Another Blues* (1965) | `kellyw-anotherblues-chamberspcobbj-1965-72c37dbf` | 0.4124 | **0.0000** | 0.6546 |
| **Wynton Kelly** - *Don't Cha Hear Me Callin'* (1966) | `kellyw-dontchahearmecallin-mcclurercobbj-1966-c450ccdb` | 0.2781 | **0.0000** | 0.0000 |
| **Wynton Kelly** - *On A Clear Day* (1966) | `kellyw-onacleardayyou-mcclurercobbj-1966-b01f94af` | 0.3662 | **0.0000** | 0.9808 |
| **Oscar Peterson** - *Hallelujah Trail* (1974) | `petersono-hallelujahtrail-orstednhannaj-1974-4c9b666d` | 0.5152 | **0.0000** | 0.6667 |
| **Bill Evans** - *Periscope* (1961) | `evansb-perisscope-lafarosmotianp-1961-1d09ff52` | 0.3288 | **0.0000** | 1.0000 |
| **Lennie Tristano** - *Movin' Along* (1956) | `tristanol-movinalong-indplevitta-1956-29226550` | 0.2716 | **0.0000** | 0.0000 |
| **Kenny Drew** - *Be My Love* (1951) | `drewk-bemylove-russellcblakeya-1951-15d2c22f` | 0.6618 | 0.8681 | **0.0000** |
| **Herbie Nichols** - *Query* (1955) | `nicholsh-query-koticktroachm-1955-7d101c19` | 0.4825 | 0.7179 | **0.0000** |
| **Herbie Nichols** - *Furthermore* (1955) | `nicholsh-furthermorealternate-mckibbonaroachm-1955-428eceab` | 0.7355 | 0.3262 | **0.0000** |
| **Herbie Nichols** - *Crisp Day* (1957) | `nicholsh-crispday-mckibbonablakeya-1957-8d73db7a` | 0.7564 | 0.8640 | **0.0000** |
| **Thelonious Monk** - *Evidence* (1971) | `monkt-evidence-mckibbonablakeya-1971-3d0e983e` | 0.4135 | 0.1476 | **0.0000** |
| **Tommy Flanagan** - *The Jumpin' Blues* (1963) | `mancej-thejumpinblues-cranshawbrokerm-1963-12545e2f` | 0.5371 | 0.8007 | **0.0000** |
| **Junior Mance** - *Softly, As in a Morning...* (1994) | `mancej-softlyasinamorning-woodejdurhamb-1994-abb8540c` | 0.4317 | 0.2239 | **0.0000** |
| **Junior Mance** - *Good Bait* (1997) | `mancej-goodbait-thompsondalleynea-1997-d4b26a19` | 0.7134 | 0.9723 | **0.0000** |
| **Bill Evans** - *Make Someone Happy* (1966) | `evansb-makesomeonehappy-israelscbunkerl-1966-68b2479f` | 0.3867 | 0.3247 | **0.0000** |

---

## 3. Notable Insights

1. **Expressive Performance Challenges**:
   - **Keith Jarrett**'s *I Thought About You* and **McCoy Tyner**'s *Darn That Dream* are the toughest beat-tracking songs overall across all models. Rubato tempo modulation and subtle brush drum styles cause high uncertainty in prediction.
2. **Impact of Fine-tuning**:
   - On syncopated swing classics like **Bud Powell's *Salt Peanuts*** or **Oscar Peterson's *Woody'n You***, fine-tuning on JTD completely solved beat detection issues, bringing them from a low `0.66` up to a perfect `1.0`.
3. **Downbeat tracking is still an open problem**:
   - Both the Baseline and Madmom models score exactly `0.0000` downbeats on their worst tracks, which represent extreme syncopation or tempo drift. Fine-tuning on JTD helps elevate scores to positive ranges (`0.20`–`0.40`) but highlights that tracking downbeats in modern jazz trio settings remains a difficult challenge.

---

## 4. Qualitative Listening Analysis

To complement the quantitative results, we conducted a qualitative audit by listening to click-track overlays of the two lowest-scoring tracks across all three models:

| Track | Fine-Tuned BT | Baseline BT | Madmom |
| :--- | :---: | :---: | :---: |
| **McCoy Tyner** — *Darn That Dream* (1989) | 0.531 | 0.468 | 0.504 |
| **Keith Jarrett** — *I Thought About You* (1993) | 0.619 | 0.375 | 0.412 |

Click encoding: **1000 Hz** = beats &nbsp;|&nbsp; **500 Hz** = downbeats

### Recurring Error Categories

#### 1. Sparse Trio Textures and Drum Drop-outs

Models frequently lose track of the pulse during passages where the drummer reduces explicit timekeeping or temporarily drops out. In these sections the beat may be implied by bass motion or harmonic rhythm rather than by clear percussive onsets. This ambiguity is especially acute when the double bass shifts into triplet figures, double-time runs, or other sub-metrical gestures. Systems trained predominantly on recordings with more explicit rhythmic textures tend to lose confidence or drift toward a competing pulse when percussion recedes.

#### 2. Swing Timing and Subdivision Ambiguity

Swung eighth notes are not evenly spaced, which creates systematic ambiguity between beat-level and subdivision-level cues. Models may interpret strong off-beat articulations as beat candidates, producing phase errors that are compounded when piano comping emphasises syncopated positions. This manifests as a systematic one-eighth-note phase offset in the predicted beat sequence.

#### 3. Walking Bass vs. Melodic Comping

A steady walking bass generally reinforces the quarter-note pulse and aids beat tracking. However, expressive articulation, anticipations, and passing tones produce onset patterns that do not align perfectly with annotated beats. This effect is especially audible at the opening of *I Thought About You*, where the bassist plays a melodic, non-walking line rather than a conventional quarter-note pattern—a texture that appears to disorient all three models early in the track.

#### 4. Systematic Miss of Beat 2

In both tracks, the fine-tuned and baseline Beat This! models produce clicks that are largely on the beat yet systematically skip beat 2. The baseline additionally introduces a small number of false positives (double clicks), suggesting it retains slightly less temporal precision than the fine-tuned variant.

#### 5. Half-Time / Double-Time Confusion

On *Darn That Dream*, Madmom outperforms both Beat This! variants despite its lower overall F-measure rank. The piece maintains a more consistent rhythmic texture, which appears to favour Madmom's DBN-based inference. By contrast, the baseline Beat This! output alternates between half-time and double-time tracking—a class of error that is reflected quantitatively by AML metrics exceeding CML metrics.

### Downbeat Tracking

Downbeat tracking remains substantially harder than beat tracking across all models. Jazz performances frequently avoid explicit downbeat reinforcement, particularly during solos, and bar positions are often implied rather than articulated. Even when the beat sequence is locally correct, the model may assign an incorrect bar position. This structural difficulty explains why fine-tuning on JTD improved beat-tracking scores while downbeat-tracking scores declined: the model has learned the local pulse of jazz trio performance more effectively than its larger-scale bar structure.
