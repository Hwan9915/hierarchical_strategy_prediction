STRATEGY_KEYWORDS = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]

STAGE_KEYWORDS = {
    0: "Exploration",
    1: "Exploration",
    2: "Comforting",
    3: "Comforting",
    4: "Comforting",
    5: "Action",
    6: "Action",
    7: "Others",
}

STRATEGY_DICT = {
    "Question": 0,
    "Restatement or Paraphrasing": 1,
    "Reflection of feelings": 2,
    "Self-disclosure": 3,
    "Affirmation and Reassurance": 4,
    "Providing Suggestions": 5,
    "Information": 6,
    "Others": 7,
}

LABEL_ABBREV = {
    "Exploration": "ex",
    "Supporting": "su",
    "Comforting": "co",
    "Action": "ac",
    "Others": "ot",
    "Question": "qu",
    "Restatement or Paraphrasing": "rp",
    "Reflection of feelings": "rf",
    "Reflection of Feelings": "rf",
    "Self-disclosure": "sd",
    "Affirmation and Reassurance": "ar",
    "Providing Suggestions": "ps",
    "Information": "in",
}

MODULE_DICT = {
    "exploration_support_identifier": [2, ["Exploration", "Supporting"], {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}, 0],
    "comforting_action_identifier": [3, ["Comforting", "Action", "Others"], {0: -1, 1: -1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2}, 0],
    "exploration_predictor": [2, ["Question", "Restatement or Paraphrasing"], {0: 0, 1: 1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1}, 1],
    "comforting_predictor": [3, ["Reflection of feelings", "Self-disclosure", "Affirmation and Reassurance"], {0: -1, 1: -1, 2: 0, 3: 1, 4: 2, 5: -1, 6: -1, 7: -1}, 1],
    "action_predictor": [2, ["Providing Suggestions", "Information"], {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: 0, 6: 1, 7: -1}, 1],
    "test": [8, list(STRATEGY_DICT.keys()), {i: i for i in range(8)}, 0],
}
