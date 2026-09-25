# 2026 Literature and Novelty Update

## Positioning the paper

The final paper should not claim to be the first quantitative goalkeeper evaluation, the first pass-risk model, or the first football action-value framework. Its distinctive contribution is the combination of a goalkeeper-specific two-branch ex-ante valuation, explicit opponent-side failure consequence, cross-environment separation of probability portability from consequence portability, selected-sample target identification, and ablation showing that finer failure-cost granularity does not necessarily earn predictive complexity.

## Key literature

- **Lamas et al. (2018)** develop a probabilistic method for separating goalkeeper decision quality from action outcome in a positioning application. This is an important conceptual predecessor. DOI: 10.1371/journal.pone.0191431.
- **Yam (2019, MIT Sloan)** proposes a broad goalkeeper evaluation framework including distribution, attacking contribution and response to pressure. The present paper is narrower and focuses on ex-ante distribution decisions under asymmetric failure.
- **Power et al. (2017)** separate pass risk and reward using tracking data. DOI: 10.1145/3097983.3098051.
- **Decroos et al. (2019)** introduce VAEP for context-aware action valuation. DOI: 10.1145/3292500.3330758.
- **Fernandez, Bornn & Cervone (2021)** develop a fine-grained expected possession value framework. DOI: 10.1007/s10994-021-05989-6.
- **Anzer & Bauer (2022)** model pass difficulty using spatiotemporal data. DOI: 10.1007/s10618-021-00810-3.
- **Kielkopf & Keiner (2025)** show distribution is roughly 73-77% of goalkeeper actions across youth and professional samples, strengthening the practical importance of in-possession analysis. DOI: 10.1177/17479541241275914.
- **van Arem et al. (2026)** explicitly argue that xT model quality must be quantified before scouting application, reinforcing the need for criterion validation and reproducibility. arXiv:2604.21087.
- **Le Coz et al. (2026)** propose a semi-Markov temporal expected-threat framework, illustrating continued development of football possession-value methods. Scientific Reports 16:22590.
- **Ding et al. (2026)** analyze 29,911 WSL goalkeeper passes using graph-based clustering and show the importance of spatial context, pressure and possession phase. DOI: 10.1080/02640414.2026.2711550.
- **Analytics FC (2023)** provides an applied goalkeeper risk-versus-reward analysis using retention, xG/xGA and xT. It is useful practitioner prior art and should be acknowledged rather than ignored.
- **FIFA's distribution framework** (Around, Through, Into, Onto, Beyond) and **UEFA Goalkeeper Coaching** give the coach-language bridge used in the practitioner article.

## Novelty wording recommended

> The contribution is not a claim to invent goalkeeper analytics or pass valuation. It is a goalkeeper-specific identification and validation study of distribution as a two-branch decision: a context-dependent probability of retention combined with an asymmetric lost branch, tested across football environments and explicitly ablated to determine how much failure-cost granularity public event data can support.
