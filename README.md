# 🧠 FarajaMH: Generative AI for Early Mental Health Screening in Low-Resource Settings

**FarajaMH** is an experimental research project developing and evaluating a *generative AI assistant* for early screening of **anxiety, depression, and psychosis** through ordinary, natural-language conversations.
The project seeks to bridge the gap in mental health screening and referrals in **low-resource settings** such as Kenya and Tanzania, where community health workers (CHWs) often serve as the first point of contact for care but have limited tools and mental health training.

---

## 🌍 Background

In East Africa, common mental disorders (CMDs) frequently go undetected due to **shortages of mental health specialists** and limited culturally adapted screening tools.
Traditional structured instruments, while clinically validated, often miss **local idioms of distress and contextual nuances** (Marangu et al., 2021).

**FarajaMH** aims to close this gap by creating a **conversational AI model** that understands and responds to both English and Kiswahili expressions of mental distress — empowering CHWs to conduct early, empathetic, and contextually grounded screening at the community level.

---

## 🧩 Foundations and Collaborations

FarajaMH builds upon three major initiatives:

1. **INSPIRE Mental Health (INSPIRE MH)** — a Wellcome Trust–funded project that harmonized 50,000+ longitudinal mental health records from HDSS sites across Kenya, Uganda, and Tanzania using the **OMOP Common Data Model** and **DDI Lifecycle**.
   *This provides the AI-ready data backbone for FarajaMH.*

2. **Weather Events and Mental Health Analysis (WEMA)** — explored the link between climate shocks and mental health using **participatory digital storytelling** in informal settlements such as Mukuru Kwa Reuben, Kenya.
   *This strengthened community trust and destigmatized mental health dialogue.*

3. **Perinatal Mental Health Study in Korogocho** — linking screening to maternal and child outcomes, reinforcing longitudinal engagement and ethical research practices.
   *This work informs FarajaMH’s co-design with community and clinical partners.*

---

## 🧠 The FarajaMH Model

The **FarajaMH generative AI model** is fine-tuned using culturally rich, annotated data sources:

* INSPIRE MH datasets aligned to DSM-5-TR symptom mappings
* Clinical notes from **Mathari Teaching and Referral Hospital (Kenya)**
* Conversational data from the **Bonga app** (Kenya)
* CHW chat data from Kenya and Tanzania
* Voice data from the **Kisesa HDSS referral hospital (Tanzania)** capturing acoustic markers of distress

All datasets undergo **ETL harmonization** into the OMOP CDM framework, documented under **DDI Lifecycle** for transparency and FAIR compliance.

The model aims to:

* Recognize culturally specific expressions of mental distress
* Generate context-sensitive responses in **English and Kiswahili**
* Support early screening, triage, and referral decisions
* Operate safely and ethically in community and clinical environments

---

## ⚙️ System Components

| Component                           | Description                                                                                                                |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Fine-tuning pipeline**            | Uses harmonized OMOP data and annotated conversational datasets.                                                           |
| **FAISS-based RAG system**          | Enables real-time, context-aware question recommendations during live conversations.                                       |
| **Multimodal input**                | Integrates text, voice, and acoustic features for richer inference.                                                        |
| **CrewAI multi-agent architecture** | Simulates clinician–patient–listener–recommender interactions to improve realism and model learning.                       |
| **Web Application (FarajaMH Tool)** | Front-end for pilot deployment, supporting live speech-to-text, conversational screening, and dynamic feedback dashboards. |

---

## 🧪 Research Design

FarajaMH will be evaluated through a **four-step step-wedge design** across HDSS sites in Kenya and Tanzania:

1. **Model Training & Technical Validation**
   Test for accuracy, coherence, and safety using controlled datasets.
2. **Pilot Deployment in HDSS Settings**
   Supervised real-world screening with CHWs and persons with lived experience (PLEs).
3. **Usability & Acceptability Study**
   Evaluate cultural resonance, user trust, and clinical integration.
4. **Ethical & Policy Review**
   Assess implications for responsible AI adoption in African mental health systems.

Ethical oversight will be provided by a dedicated **Ethics and Trustworthy AI Committee**.

---

## 🧭 Vision and Impact

FarajaMH is more than a technical innovation — it is an **African-led, ethically governed, community-validated AI** initiative designed to:

* Enable early identification of anxiety, depression, and psychosis
* Empower CHWs through AI-supported guidance
* Reduce stigma and foster community dialogue
* Position HDSS communities as **co-creators** of responsible AI in health research

---



## 🧾 Citation

If you use FarajaMH or its methods, please cite:

> African Population and Health Research Center (APHRC). *FarajaMH: Generative AI for Early Screening of Anxiety, Depression, and Psychosis in Low-Resource Settings.* Nairobi, Kenya, 2025.

---

## 🧠 Acknowledgments

FarajaMH is led by the **Data Science Program** at the **African Population and Health Research Center (APHRC)** with support from the **Wellcome Trust GenAI Accelerator**.
Collaborating institutions include Mathari Teaching and Referral Hospital, Kisesa HDSS, and multiple HDSS sites across East Africa.

---

## ⚖️ License

This project is released under the **MIT License** for research and non-commercial use.
Ethical approval is required for any deployment involving human participants.

---

Would you like me to include a short “**Getting Started**” section with environment setup and Docker instructions — or keep this as a high-level research README?
