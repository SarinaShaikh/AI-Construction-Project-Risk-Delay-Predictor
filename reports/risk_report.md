# Phase 5 — Construction Activity Risk Assessment Report

## 1. Overview

This report documents the construction activity risk assessment performed in Phase 5 of the AI Construction project.

The risk assessment combines:

* Machine Learning predicted activity delay
* Delay probability
* CPM-based activity criticality
* Total float
* Float impact

The objective is to identify construction activities that require the highest level of attention and prioritize them for risk mitigation.

---

## 2. Risk Scoring Methodology

The overall risk score is calculated using:

```text
Risk Score =
Delay Probability × Criticality Weight × Float Impact
```

### 2.1 Delay Probability

The ML model generates a predicted delay in days for each activity.

The predicted delay is converted into a probability-like score using the 95th percentile of predicted delays as the maximum reference value.

```text
Delay Probability =
Predicted Delay / 95th Percentile Delay
```

The value is then limited to the range:

```text
0 ≤ Delay Probability ≤ 1
```

---

### 2.2 Criticality Weight

The criticality weight represents the importance of an activity based on its total float.

| Total Float | Criticality Weight |
| ----------- | -----------------: |
| ≤ 0 days    |                1.0 |
| < 5 days    |                0.9 |
| < 20 days   |                0.7 |
| < 50 days   |                0.4 |
| ≥ 50 days   |                0.1 |

Activities with little or no float receive a higher weight because delays in these activities can have a greater effect on the project schedule.

---

### 2.3 Float Impact

Float impact measures how significant the predicted delay is compared with the available scheduling flexibility.

```text
Float Impact =
Predicted Delay / max(Total Float, 1)
```

The result is limited to a maximum of 1.0.

For activities with zero float and a positive predicted delay, the float impact becomes 1.0.

---

## 3. Dataset

Risk scoring was performed for:

* **Total activities:** 9,279
* **Critical activities:** 3,153
* **Non-critical activities:** 6,126

The ML predictions were generated using the trained Gradient Boosting Regressor model from Phase 4.

---

## 4. Risk Score Results

The calculated risk scores produced the following overall statistics:

| Metric                        |  Value |
| ----------------------------- | -----: |
| Total activities              |  9,279 |
| Mean risk score               | 0.2509 |
| Risk score standard deviation | 0.2721 |
| Minimum risk score            | 0.0000 |
| Maximum risk score            | 1.0000 |
| High-risk activities          |  2,320 |
| Critical-risk activities      |    464 |

High-risk activities are activities above the 75th percentile of the risk-score distribution.

Critical-risk activities are activities above the 95th percentile.

---

## 5. Predicted Delay Results

The ML model generated predicted delay values for all 9,279 activities.

| Metric               |        Value |
| -------------------- | -----------: |
| Mean predicted delay | 17.0777 days |
| Standard deviation   |  9.6153 days |

The predicted delays are used as one of the main inputs to the risk scoring process.

---

## 6. Critical Path Risk Analysis

CPM results were used to distinguish critical-path activities from non-critical activities.

### Critical Path

* Activities: **3,153**
* Average risk score: **0.4934**
* Maximum risk score: **1.0000**
* Average predicted delay: **17.2818 days**

### Non-Critical Path

* Activities: **6,126**
* Average risk score: **0.1260**
* Maximum risk score: **0.9000**
* Average predicted delay: **16.9726 days**

The results show that critical-path activities have a substantially higher average risk score than non-critical activities.

This indicates that combining ML-predicted delays with CPM criticality provides a useful way to prioritize activities that could have a stronger effect on project completion.

---

## 7. Highest-Risk Activities

The risk-scoring system ranks every activity according to its calculated risk score.

The highest-risk activities include:

| Risk Rank | Activity ID  | Project ID | Risk Score |
| --------: | ------------ | ---------- | ---------: |
|         1 | P00001_A0021 | P00001     |     1.0000 |
|         2 | P00001_A0055 | P00001     |     1.0000 |
|         3 | P00002_A0001 | P00002     |     1.0000 |
|         4 | P00002_A0081 | P00002     |     1.0000 |
|         5 | P00003_A0011 | P00003     |     1.0000 |
|         6 | P00003_A0013 | P00003     |     1.0000 |
|         7 | P00003_A0059 | P00003     |     1.0000 |
|         8 | P00003_A0073 | P00003     |     1.0000 |
|         9 | P00005_A0019 | P00005     |     1.0000 |
|        10 | P00005_A0079 | P00005     |     1.0000 |

These activities should receive priority during construction monitoring and risk mitigation.

---

## 8. Risk Analysis by Criticality

The analysis demonstrates an important relationship between CPM criticality and risk.

Critical-path activities have an average risk score of approximately:

```text
0.4934
```

while non-critical activities have an average risk score of approximately:

```text
0.1260
```

Therefore, critical-path activities have considerably higher average risk under the implemented scoring methodology.

This supports the use of CPM information together with machine-learning predictions instead of considering predicted delay alone.

---

## 9. Risk Analysis by Phase and Resource

The risk analysis notebook also evaluates risk according to:

* Construction phase
* Resource type
* Project
* Critical-path status

These analyses help identify areas where higher predicted delays or higher risk concentrations occur.

The generated notebook contains visualizations for:

* Risk-score distribution
* Risk categories
* Top 20 highest-risk activities
* Average risk by construction phase
* Average predicted delay by construction phase
* Critical vs non-critical risk
* Delay prediction vs risk score
* Project-level risk
* Resource-type risk
* High-risk activities by phase

---

## 10. Generated Outputs

Phase 5 generated the following important outputs:

### Risk Scores

```text
data/processed/risk_scores.csv
```

This file contains the calculated risk score and ranking for all activities.

### Analyzed Risk Dataset

```text
data/processed/risk_scores_analyzed.csv
```

This file contains the enriched risk dataset used for detailed analysis.

### Risk Analysis Notebook

```text
notebooks/05_risk_analysis.ipynb
```

The notebook contains the statistical analysis and visualizations used to interpret the risk results.

---

## 11. Key Findings

The Phase 5 analysis produced the following findings:

1. The system successfully generated risk scores for all **9,279 construction activities**.

2. The mean overall risk score was **0.2509**.

3. **2,320 activities** were classified as high-risk using the 75th-percentile threshold.

4. **464 activities** were classified as critical-risk using the 95th-percentile threshold.

5. **3,153 activities** were identified as critical-path activities using the CPM results.

6. Critical-path activities had an average risk score of **0.4934**, compared with **0.1260** for non-critical activities.

7. The average predicted delay across activities was approximately **17.08 days**.

8. The risk-ranking system allows construction activities to be prioritized based on both predicted delay and schedule impact.

---

## 12. Interpretation

The Phase 5 risk assessment combines two complementary approaches.

Machine learning estimates the potential delay associated with an activity, while CPM determines how much scheduling flexibility the activity has.

An activity with a large predicted delay but substantial float may have lower overall project risk.

Conversely, an activity with a predicted delay and little or no float can receive a high risk score because there is less schedule flexibility available.

This combination provides a more practical risk-prioritization mechanism for construction project management.

---

## 13. Limitations

The current risk score is a project-risk prioritization metric rather than a calibrated probability of project failure.

The delay probability is derived by normalizing ML predictions rather than using a separately calibrated probabilistic model.

The risk score also depends on the selected criticality-weight thresholds and float-impact formula.

Future versions can improve the methodology using:

* Probability calibration
* Time-series project monitoring
* Real-time activity status
* Weather information
* Material delivery delays
* Labour availability
* Equipment availability
* Historical project outcomes
* More advanced ML models

---

## 14. Conclusion

Phase 5 successfully integrates the machine-learning delay prediction model with CPM scheduling information to produce an activity-level construction risk assessment.

The resulting system ranks activities according to their predicted delay, schedule criticality, and available float.

The analysis identified **464 critical-risk activities** and **2,320 high-risk activities** among **9,279 activities**.

The substantially higher average risk observed for critical-path activities demonstrates the value of combining machine-learning predictions with traditional construction scheduling analysis.

The generated risk scores provide a foundation for the next stage of the AI Construction system, including risk visualization, project-level decision support, and integration into a construction management dashboard.
