## 1. Исследовательский обзор

**Тема ВКР:** разработка системы автоматической разметки музыкального каталога по типу фронт-вокала для задач поиска и рекомендаций.

**Постановка ЛР3.** Разметка поступающего потока треков формулируется как кооперативная многоагентная задача принятия решений. Три агента-эксперта соответствуют модальностям из ЛР2 (текст метаданных, временной ряд звука, спектрограмма): они голосуют за класс вокала или воздерживаются. Агент-координатор решает, поставить метку автоматически или отправить трек модератору. Бюджет модерации ограничен, ошибка в каталоге дорого стоит (портит поиск и рекомендации), а ручная проверка играет роль транзакционных издержек.

**Исследовательский вопрос:** даёт ли согласованная многоагентная политика выигрыш по итоговой ценности разметки (качество каталога минус издержки) по сравнению с простым правилом и с одноагентной политикой, и сохраняется ли он на отложенном периоде и другом каталоге?

**Поисковый запрос** (Google Scholar, arXiv, NeurIPS / ICML / ICLR / AAAI, 2017–2026):
`("multi-agent reinforcement learning" OR MARL) AND (cooperative OR "credit assignment" OR CTDE OR "constrained" OR "safe") ; ("learning to defer" OR "active learning") AND "reinforcement learning"`

### Аннотированный обзор

| № | Источник | Группа | Суть | Значение для работы |
|---|---|---|---|---|
| 1 | Lowe R. et al. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG).* NeurIPS 2017. [arXiv:1706.02275](https://arxiv.org/abs/1706.02275) | CTDE, actor-critic | централизованный критик на совместных действиях, децентрализованные акторы | алгоритм MADDPG (у нас дискретный вариант с Gumbel-softmax) |
| 2 | Foerster J. et al. *Counterfactual Multi-Agent Policy Gradients (COMA).* AAAI 2018. [arXiv:1705.08926](https://arxiv.org/abs/1705.08926) | credit assignment | контрфактический baseline: вклад агента = Q(s,a) − ожидание по его альтернативным действиям | алгоритм COMA; идея оценки вклада агентов |
| 3 | Sunehag P. et al. *Value-Decomposition Networks (VDN).* AAMAS 2018. [arXiv:1706.05296](https://arxiv.org/abs/1706.05296) | value decomposition | Q_tot = сумма индивидуальных Q | алгоритм VDN |
| 4 | Rashid T. et al. *QMIX.* ICML 2018. [arXiv:1803.11485](https://arxiv.org/abs/1803.11485) | value decomposition, CTDE | монотонное смешивание Q агентов гиперсетью от глобального состояния | алгоритм QMIX |
| 5 | de Witt C. S. et al. *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?* 2020. [arXiv:2011.09533](https://arxiv.org/abs/2011.09533) | независимое обучение | IPPO без централизованного критика конкурентоспособен с CTDE | алгоритм IPPO; гипотеза «независимое обучение достаточно» |
| 6 | Yu C. et al. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO).* NeurIPS 2022. [arXiv:2103.01955](https://arxiv.org/abs/2103.01955) | CTDE, on-policy | PPO с централизованным критиком — сильный baseline; протокол сравнения по seed | алгоритм MAPPO; протокол |
| 7 | Kuba J. G. et al. *Trust Region Policy Optimisation in Multi-Agent RL (HATRPO/HAPPO).* ICLR 2022. [arXiv:2109.11251](https://arxiv.org/abs/2109.11251) | CTDE, гарантии улучшения | последовательное обновление агентов с монотонным улучшением | контекст; не реализован (ограничение) |
| 8 | Wen M. et al. *Multi-Agent Reinforcement Learning is a Sequence Modeling Problem (MAT).* NeurIPS 2022. [arXiv:2205.14953](https://arxiv.org/abs/2205.14953) | трансформеры | совместное действие как последовательность | контекст; не реализован (ограничение) |
| 9 | Achiam J. et al. *Constrained Policy Optimization.* ICML 2017. [arXiv:1705.10528](https://arxiv.org/abs/1705.10528) | безопасный RL | RL с ограничениями на ожидаемые издержки | ограничение бюджета модерации и SLA ошибок |
| 10 | Gu S. et al. *Multi-Agent Constrained Policy Optimisation (MACPO).* 2021. [arXiv:2110.02793](https://arxiv.org/abs/2110.02793) | безопасный MARL | MARL с ограничениями безопасности | обоснование учёта нарушений ограничений как отдельного критерия |
| 11 | Papoudakis G. et al. *Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks (EPyMARL).* NeurIPS 2021 Datasets & Benchmarks. [arXiv:2006.07869](https://arxiv.org/abs/2006.07869) | бенчмарк | единое сравнение IQL, VDN, QMIX, COMA, MADDPG, MAA2C, MAPPO, IPPO | ровно наш набор из 8 алгоритмов и протокол |
| 12 | Terry J. K. et al. *PettingZoo: Gym for Multi-Agent RL.* NeurIPS 2021. [arXiv:2009.14471](https://arxiv.org/abs/2009.14471) | среды | стандарт API многоагентных сред | интерфейс среды (obs / action mask / state) |
| 13 | Liu X.-Y. et al. *FinRL: A Deep RL Library for Automated Stock Trading.* NeurIPS 2020 Deep RL Workshop. [arXiv:2011.09607](https://arxiv.org/abs/2011.09607) | финансовые приложения | RL с транзакционными издержками и рисковыми метриками | перенос финансовых метрик (Sharpe, CVaR, просадка) на разметку |
| 14 | Скрынник А. и др. (AIRI, Россия). *POGEMA: A Benchmark Platform for Cooperative Multi-Agent Pathfinding.* 2024. [arXiv:2407.14931](https://arxiv.org/abs/2407.14931); *Learn to Follow: Decentralized Lifelong Multi-agent Pathfinding via Planning and Learning.* AAAI 2024. [arXiv:2310.01207](https://arxiv.org/abs/2310.01207) | российские результаты | открытая платформа и протокол оценки кооперативного MARL; гибрид планирования и MARL | отечественный пример воспроизводимого протокола; при децентрализации важна простая коммуникация |
| 15 | Madras D., Pitassi T., Zemel R. *Predict Responsibly: Learning to Defer.* NeurIPS 2018. [arXiv:1711.06664](https://arxiv.org/abs/1711.06664); Fang M., Li Y., Cohn T. *Learning how to Active Learn: A Deep RL Approach.* EMNLP 2017. [arXiv:1708.02383](https://arxiv.org/abs/1708.02383) | человек в контуре | модель учится, когда передать решение эксперту; RL-политика выбора объектов для разметки | прямая основа роли агента-координатора (автометка или модератор) |

### Сравнительная матрица

| Алгоритм / работа | Класс | Критик | Credit assignment | Действия | Ограничения | Среды в статье | Код |
|---|---|---|---|---|---|---|---|
| IQL | независимый value-based | — | нет | дискретные | нет | SMAC, LBF | EPyMARL |
| VDN | value decomposition | сумма Q | аддитивный | дискретные | нет | 2D-игры | EPyMARL |
| QMIX | value decomposition | гиперсеть от s | монотонный | дискретные | нет | SMAC | pymarl |
| IPPO | независимый policy-gradient | V(o_i) | нет | любые | нет | SMAC | есть |
| MAPPO | CTDE policy-gradient | V(s) | общий V | любые | нет | MPE, SMAC, Hanabi | есть |
| MAA2C | CTDE actor-critic | V(s) | общий V | любые | нет | SMAC, LBF | EPyMARL |
| COMA | CTDE actor-critic | Q(s, u) | контрфактический | дискретные | нет | StarCraft | pymarl |
| MADDPG | CTDE actor-critic | Q(s, u) | централизованный Q | непрерывные (дискретные через Gumbel) | нет | MPE | есть |
| MACPO / CPO | безопасный (MA)RL | V + cost-V | — | непрерывные | да | Safe MAMuJoCo | есть |
| Learning to defer | supervised + отказ | — | — | {решить, передать} | бюджет (неявно) | классификация | есть |
| **Эта работа** | **8 MARL + PPO + правило** | **разные** | **оценка вклада агентов** | **голос / воздержание; автометка / проверка** | **бюджет модерации, SLA ошибок** | **поток треков MTAT → Jamendo** | **да** |

### Исследовательский разрыв

1. MARL-алгоритмы сравниваются на играх и робототехнике (SMAC, MPE, POGEMA). Задача **разметки контента с человеком в контуре** как кооперативной игры экспертов-модальностей и координатора в MARL-литературе не описана.
2. Работы о learning-to-defer (15) рассматривают одного классификатора и не учитывают бюджет проверок в эпизоде и динамику доверия к экспертам.
3. Финансовые работы (13) вводят метрики риска (Sharpe, CVaR, просадка), но их не применяют к операционной задаче качества каталога.

### Гипотезы

- **H1.** Лучший MARL-алгоритм даёт среднюю ценность разметки выше эвристики «равные веса + порог» на отложенном тесте MTAT (интервал по seed не включает разницу 0).
- **H2.** Алгоритмы с централизованным обучением (MAPPO, QMIX) превосходят независимые (IQL, IPPO).
- **H3.** Многоагентная декомпозиция не хуже одноагентного PPO с полным состоянием.
- **H4.** Выигрыш сохраняется на другом каталоге и периоде (Jamendo 2010–2017).
