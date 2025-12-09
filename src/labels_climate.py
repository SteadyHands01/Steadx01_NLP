# Label mapping for CLIMATE fallacy dataset

FALLACY_LABELS = [
    "CHERRY_PICKING",
    "EVADING_THE_BURDEN_OF_PROOF",
    "FALSE_ANALOGY",
    "FALSE_AUTHORITY",
    "FALSE_CAUSE",
    "HASTY_GENERALISATION",
    "NO_FALLACY",
    "POST_HOC",
    "RED_HERRINGS",
    "STRAWMAN",
    "VAGUENESS",
]

LABEL2ID = {lbl: i for i, lbl in enumerate(FALLACY_LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
